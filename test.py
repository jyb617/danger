#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 realtime.py 实时推理功能
"""

import cv2
import time
import argparse
from inferences.realtime import RealtimeInferenceSession


def test_realtime_basic(video_source):
    """基础实时推理测试"""
    print("\n" + "=" * 60)
    print("测试实时推理基础功能")
    print("=" * 60 + "\n")

    print(f"视频源: {video_source}")

    # 创建实时推理会话
    print("\n1. 创建实时推理会话...")
    try:
        session = RealtimeInferenceSession(video_source)
        print("✓ 会话创建成功")
    except Exception as e:
        print(f"❌ 会话创建失败: {e}")
        return

    # 等待初始化
    print("\n2. 等待会话初始化...")
    time.sleep(2)
    print("✓ 初始化完成")

    # 测试获取结果
    print("\n3. 测试获取实时结果...")
    success_count = 0
    for i in range(10):
        result = session.get_result()
        if result is not None:
            success_count += 1
            print(f"  帧 {i + 1}: ✓ 获取成功 (形状: {result.shape})")
        else:
            print(f"  帧 {i + 1}: ⚠ 暂无结果")
        time.sleep(0.5)

    print(f"\n✓ 成功获取 {success_count}/10 帧")

    # 释放会话
    print("\n4. 释放会话...")
    session.release()
    print("✓ 会话已释放")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60 + "\n")


def test_realtime_display(video_source):
    """实时显示测试"""
    print("\n" + "=" * 60)
    print("测试实时推理显示")
    print("=" * 60 + "\n")

    print(f"视频源: {video_source}")
    print("按 'q' 键退出\n")

    # 创建实时推理会话
    session = RealtimeInferenceSession(video_source)

    # 等待初始化
    time.sleep(2)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            result = session.get_result()

            if result is not None:
                # 显示结果
                cv2.imshow('Realtime Inference - Press Q to quit', result)

                frame_count += 1

                # 每秒输出一次帧率
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"帧数: {frame_count}, FPS: {fps:.2f}")

            # 检查退出键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.033)  # 约30fps

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        # 清理
        cv2.destroyAllWindows()
        session.release()

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        print(f"\n统计信息:")
        print(f"  总帧数: {frame_count}")
        print(f"  总时间: {elapsed:.2f}秒")
        print(f"  平均FPS: {fps:.2f}")

        print("\n测试完成！\n")


def test_realtime_save(video_source, output_path, duration=10):
    """保存实时推理结果"""
    print("\n" + "=" * 60)
    print("测试保存实时推理结果")
    print("=" * 60 + "\n")

    print(f"视频源: {video_source}")
    print(f"输出路径: {output_path}")
    print(f"持续时间: {duration}秒\n")

    # 创建实时推理会话
    session = RealtimeInferenceSession(video_source)

    # 等待初始化
    time.sleep(2)

    # 获取第一帧以确定视频参数
    result = session.get_result()
    while result is None:
        time.sleep(0.1)
        result = session.get_result()

    height, width = result.shape[:2]
    fps = 30

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"视频参数: {width}x{height} @ {fps}fps")
    print(f"开始录制...\n")

    frame_count = 0
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            result = session.get_result()

            if result is not None:
                writer.write(result)
                frame_count += 1

                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    print(f"已录制: {elapsed:.1f}秒 ({frame_count} 帧)")

            time.sleep(0.033)

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        writer.release()
        session.release()

        elapsed = time.time() - start_time
        print(f"\n✓ 录制完成:")
        print(f"  输出文件: {output_path}")
        print(f"  总帧数: {frame_count}")
        print(f"  总时间: {elapsed:.2f}秒")
        print(f"  平均FPS: {frame_count / elapsed:.2f}")

        print("\n测试完成！\n")


def main():
    parser = argparse.ArgumentParser(description="测试实时推理功能")
    parser.add_argument("--source", type=str, default="test_video.mp4",
                        help="视频源（文件路径或摄像头索引）")
    parser.add_argument("--mode", type=str, default="basic",
                        choices=["basic", "display", "save"],
                        help="测试模式")
    parser.add_argument("--output", type=str, default="realtime_result.mp4",
                        help="输出视频路径（仅save模式）")
    parser.add_argument("--duration", type=int, default=10,
                        help="录制时长（秒，仅save模式）")

    args = parser.parse_args()

    # 处理视频源（如果是数字则转为摄像头索引）
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source

    if args.mode == "basic":
        test_realtime_basic(video_source)
    elif args.mode == "display":
        test_realtime_display(video_source)
    elif args.mode == "save":
        test_realtime_save(video_source, args.output, args.duration)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()