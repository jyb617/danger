#!/usr/bin/env python3
"""
为现有视频重新生成封面图片
用于修复丢失的封面文件
"""

import os
import glob
import cv2
import toml

# 加载配置
configs = toml.load('servers/configs/config.toml')
cover_width = configs['cover-width']
cover_height = configs['cover-height']


def generate_cover_for_video(video_path, cover_path):
    """为视频生成封面"""
    if os.path.exists(cover_path):
        print(f"⏭  封面已存在，跳过: {cover_path}")
        return True

    if not os.path.exists(video_path):
        print(f"❌ 视频不存在: {video_path}")
        return False

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        capture.release()
        return False

    read_success, image = capture.read()
    capture.release()

    if not read_success:
        print(f"❌ 无法读取视频首帧: {video_path}")
        return False

    try:
        resized_image = cv2.resize(image, (cover_width, cover_height),
                                   interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(cover_path, resized_image)
        print(f"✓ 生成封面: {cover_path}")
        return True
    except Exception as e:
        print(f"❌ 生成封面失败: {cover_path}, 错误: {e}")
        return False


def main():
    print("=" * 60)
    print("为现有视频重新生成封面")
    print("=" * 60)

    # 查找所有视频文件
    video_patterns = [
        'servers/videos/result.*.mp4',
        'servers/videos/result.*.avi',
    ]

    video_files = []
    for pattern in video_patterns:
        video_files.extend(glob.glob(pattern))

    if not video_files:
        print("\n❌ 未找到任何视频文件")
        return

    print(f"\n找到 {len(video_files)} 个视频文件\n")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for video_path in video_files:
        # 提取视频ID
        basename = os.path.basename(video_path)
        # result.7386603832534368256.mp4 -> 7386603832534368256
        video_id = basename.replace('result.', '').rsplit('.', 1)[0]

        cover_path = f'servers/covers/result.{video_id}.jpg'

        print(f"\n处理视频: {video_id}")
        print(f"  视频文件: {video_path}")
        print(f"  封面路径: {cover_path}")

        result = generate_cover_for_video(video_path, cover_path)

        if result:
            if os.path.exists(cover_path):
                file_size = os.path.getsize(cover_path)
                if file_size > 0:
                    success_count += 1
                else:
                    skip_count += 1
            else:
                skip_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"✓ 成功生成: {success_count}")
    print(f"⏭  跳过: {skip_count}")
    print(f"❌ 失败: {fail_count}")
    print(f"📊 总计: {len(video_files)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
