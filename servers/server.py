import queue
import contextlib
import os
import datetime
import threading

import flask
import pymongo
import cv2
import toml
import snowflake

import inferences.engines as engines
import inferences.realtime as realtime

from apscheduler.schedulers.background import BackgroundScheduler
import mimetypes
from flask import Response, request, send_file, abort

app = flask.Flask(__name__)

# ❌ 不再需要 CORS，因为 Nginx 代理使前后端同源
# from flask_cors import CORS
# CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB 上传限制

configs = toml.load('servers/configs/config.toml')

connection_uri = configs['db-connection-uri']
connection_max = configs['db-connection-max']

database = pymongo.MongoClient(connection_uri, maxpoolsize=connection_max)

scheduler = BackgroundScheduler()
scheduler.start()

realtime_sessions_lock = threading.Lock()
realtime_sessions = {}

remove_queue = queue.Queue()

frames_interval = configs['frames-interval']
remove_interval = configs['remove-interval']

video_speed = configs['video-speed']
video_width = configs['video-width']
cover_width = configs['cover-width']

video_height = configs['video-height']
cover_height = configs['cover-height']

id_generator = snowflake.SnowflakeGenerator(0)


def send_video_file(filepath):
    """发送支持 Range 请求的视频文件（流式传输）"""
    try:
        abs_filepath = os.path.abspath(filepath)

        if not os.path.exists(abs_filepath):
            print(f"✗ 文件不存在: {abs_filepath}")
            return abort(404)

        file_size = os.path.getsize(abs_filepath)

        if file_size < 1000:
            print(f"⚠ 视频文件异常小: {abs_filepath} ({file_size} bytes)")
            return abort(404)

        print(f"准备发送视频: {abs_filepath} ({file_size} bytes)")

        # 确定 MIME 类型
        if abs_filepath.endswith('.mp4'):
            mimetype = 'video/mp4'
        elif abs_filepath.endswith('.avi'):
            mimetype = 'video/x-msvideo'
        else:
            mimetype = 'application/octet-stream'

        range_header = request.headers.get('Range')

        if not range_header:
            # ✅ 没有 Range 请求，使用生成器流式传输完整文件
            print(f"发送完整文件（流式）")

            def generate():
                with open(abs_filepath, 'rb') as f:
                    chunk_size = 8192  # 8KB chunks
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk

            response = Response(generate(), mimetype=mimetype)
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Content-Length'] = str(file_size)
            return response

        # ✅ 处理 Range 请求
        print(f"Range 请求: {range_header}")

        try:
            range_str = range_header.replace('bytes=', '').strip()

            if '-' not in range_str:
                # 无效格式，返回完整文件
                def generate():
                    with open(abs_filepath, 'rb') as f:
                        chunk_size = 8192
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            yield chunk

                response = Response(generate(), mimetype=mimetype)
                response.headers['Accept-Ranges'] = 'bytes'
                response.headers['Content-Length'] = str(file_size)
                return response

            parts = range_str.split('-')
            byte_start = int(parts[0]) if parts[0] else 0
            byte_end = int(parts[1]) if parts[1] else file_size - 1

            # 验证范围
            byte_start = max(0, min(byte_start, file_size - 1))
            byte_end = max(byte_start, min(byte_end, file_size - 1))

            length = byte_end - byte_start + 1

            print(f"发送范围: {byte_start}-{byte_end}/{file_size} ({length} bytes)")

            # ✅ 使用生成器流式读取指定范围
            def generate_range():
                with open(abs_filepath, 'rb') as f:
                    f.seek(byte_start)
                    remaining = length
                    chunk_size = 8192

                    while remaining > 0:
                        read_size = min(chunk_size, remaining)
                        chunk = f.read(read_size)
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            response = Response(
                generate_range(),
                206,
                mimetype=mimetype,
                direct_passthrough=True
            )

            response.headers['Content-Range'] = f'bytes {byte_start}-{byte_end}/{file_size}'
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Content-Length'] = str(length)

            return response

        except (ValueError, IndexError) as e:
            print(f"❌ 解析 Range 失败: {e}")
            return abort(400)

    except Exception as e:
        print(f"❌ send_video_file 异常: {e}")
        import traceback
        traceback.print_exc()
        return abort(500)


def save_video_cover(source, output):
    """保存视频封面"""
    capture = cv2.VideoCapture(source)

    if capture.isOpened():
        read_success, image = capture.read()

        if read_success:
            cv2.imwrite(output, cv2.resize(image, (cover_width, cover_height), interpolation=cv2.INTER_LINEAR))

    capture.release()


def save_detection_result(source, output, scores):
    """保存检测结果视频"""
    reader = cv2.VideoCapture(source)

    if not reader.isOpened():
        raise Exception(f"无法打开源视频: {source}")

    # 获取视频属性
    fps = reader.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = video_speed

    total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"源视频信息: FPS={fps}, 总帧数={total_frames}")

    # ✅ 尝试多种编码器（优先使用浏览器兼容的H.264编码）
    codecs_to_try = [
        # H.264编码器（浏览器兼容性最好）
        ('avc1', output, 'video/mp4'),
        ('H264', output, 'video/mp4'),
        ('X264', output, 'video/mp4'),
        ('h264', output, 'video/mp4'),
        ('x264', output, 'video/mp4'),
        # MPEG-4编码器（备选）
        ('mp4v', output, 'video/mp4'),
        # MJPEG编码器（浏览器支持较好）
        ('MJPG', output.replace('.mp4', '.avi'), 'video/x-msvideo'),
        # XVID编码器（最后备选）
        ('XVID', output.replace('.mp4', '.avi'), 'video/x-msvideo'),
    ]

    writer = None
    final_output = None

    for codec, path, mime in codecs_to_try:
        print(f"尝试编码器: {codec} -> {path}")
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_writer = cv2.VideoWriter(path, fourcc, fps, (video_width, video_height))

            if test_writer.isOpened():
                writer = test_writer
                final_output = path
                print(f"✓ 成功创建写入器: {codec}")
                print(f"  输出文件: {final_output}")
                print(f"  编码格式: {codec} ({'浏览器兼容' if codec in ['avc1', 'H264', 'X264', 'h264', 'x264', 'MJPG'] else '可能不兼容浏览器'})")
                break
            else:
                test_writer.release()
                print(f"✗ {codec} 编码器失败 - VideoWriter未能打开")
        except Exception as e:
            print(f"✗ {codec} 编码器异常: {e}")

    if writer is None:
        reader.release()
        raise Exception("所有视频编码器均失败")

    # 写入帧
    frame_count = 0
    write_errors = 0

    print(f"开始写入 {len(scores)} 帧...")

    for idx, score in enumerate(scores):
        read_success, frame = reader.read()

        if not read_success:
            print(f"⚠ 第 {idx} 帧读取失败")
            break

        try:
            # 调整尺寸
            if frame.shape[:2] != (video_height, video_width):
                frame = cv2.resize(frame, (video_width, video_height))

            # 绘制检测结果
            result_frame = engines.draw_detection_result(frame, score)

            # 写入帧
            write_success = writer.write(result_frame)

            if write_success is False:
                write_errors += 1
                if write_errors > 10:
                    print(f"❌ 写入失败次数过多，中止")
                    break

            frame_count += 1

            # 每100帧输出进度
            if frame_count % 100 == 0:
                print(f"进度: {frame_count}/{len(scores)} 帧")

        except Exception as e:
            print(f"❌ 处理第 {idx} 帧时出错: {e}")
            write_errors += 1

    # 释放资源
    reader.release()
    writer.release()

    print(f"✓ 完成写入 {frame_count} 帧 (共 {len(scores)} 帧预期)")

    # ✅ 验证输出文件
    if not os.path.exists(final_output):
        raise Exception(f"输出文件不存在: {final_output}")

    file_size = os.path.getsize(final_output)
    print(f"✓ 输出文件: {final_output}")
    print(f"✓ 文件大小: {file_size / (1024 * 1024):.2f} MB")

    if file_size < 10000:  # 小于10KB肯定有问题
        raise Exception(f"输出文件异常小 ({file_size} bytes)，可能损坏")

    # ✅ 尝试重新打开验证
    verify_cap = cv2.VideoCapture(final_output)
    if not verify_cap.isOpened():
        verify_cap.release()
        raise Exception("生成的视频文件无法打开")

    verify_frame_count = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    verify_fps = verify_cap.get(cv2.CAP_PROP_FPS)
    verify_fourcc = int(verify_cap.get(cv2.CAP_PROP_FOURCC))
    verify_cap.release()

    print(f"✓ 验证成功:")
    print(f"  帧数: {verify_frame_count}")
    print(f"  FPS: {verify_fps}")
    print(f"  编码: {verify_fourcc}")

    if verify_frame_count == 0:
        raise Exception("生成的视频文件帧数为0")

    remove_queue.put(source)
    return final_output


def get_realtime_data(session):
    """获取实时检测数据"""
    realtime_frame = session.get_result()

    if realtime_frame is not None:
        encode_success, encoded_frame = cv2.imencode('.jpg', realtime_frame)

        if encode_success:
            return encoded_frame.tobytes()


def generate_realtime_response(session):
    """生成实时检测响应流"""
    while True:
        response_bytes = realtime.execute_task_in_seconds(get_realtime_data, session, frames_interval)

        if response_bytes is not None:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + response_bytes + b'\r\n'


def release_and_delete_session(session_id):
    """释放并删除会话"""
    if session_id in realtime_sessions:
        session = realtime_sessions[session_id]

        session.release()
        realtime_sessions.pop(session_id)


@scheduler.scheduled_job(trigger='interval', seconds=remove_interval)
def remove_task():
    """定时删除文件任务"""
    remove_paths = []

    while not remove_queue.empty():
        remove_paths.append(remove_queue.get())

    for remove_path in remove_paths:
        try:
            with contextlib.suppress(FileNotFoundError):
                os.remove(remove_path)
                print(f"✓ 已删除文件: {remove_path}")
        except OSError as e:
            print(f"⚠ 删除文件失败: {remove_path}, 错误: {e}")
            remove_queue.put(remove_path)


@app.post('/api/videoinference')
def video_inference():
    """视频推理接口"""
    print(f"收到请求: {flask.request.method} {flask.request.path}")
    print(f"请求来源: {flask.request.remote_addr}")

    video_id = str(next(id_generator))

    video_source = f'servers/videos/source.{video_id}.mp4'
    video_output = f'servers/videos/result.{video_id}.mp4'
    cover_output = f'servers/covers/result.{video_id}.jpg'

    # 保存上传的视频
    try:
        flask.request.files['video'].save(video_source)
        print(f"✓ 视频文件已保存: {video_source}")
    except KeyError:
        print("❌ 缺少视频文件")
        return flask.abort(400, description="缺少视频文件")

    # 获取参数
    try:
        name = flask.request.form['name']
        note = flask.request.form['note']
        print(f"参数 - 名称: {name}, 备注: {note}")
    except KeyError as e:
        print(f"❌ 缺少必要参数: {e}")
        return flask.abort(400, description="缺少必要参数")

    # 保存视频封面
    save_video_cover(video_source, cover_output)
    print(f"✓ 视频封面已保存: {cover_output}")

    # 进行异常检测
    try:
        print(f"开始检测视频: {video_id}")
        scores = engines.detection_by_video(video_source).tolist()
        print(f"检测完成: {len(scores)} 个分数")
    except (ValueError, Exception) as e:
        print(f"❌ 视频检测失败: {e}")
        return flask.abort(400, description=f"视频检测失败: {str(e)}")

    # 扩展分数并保存结果视频
    try:
        expanded_scores = engines.expand_scores(scores)
        save_detection_result(video_source, video_output, expanded_scores)
        print(f"✓ 检测结果视频已保存: {video_output}")
    except Exception as e:
        print(f"❌ 保存结果视频失败: {e}")
        return flask.abort(500, description=f"保存结果失败: {str(e)}")

    # 保存到数据库
    try:
        database.surveillance.videos.insert_one({
            'videoId': video_id,
            'name': name,
            'note': note,
            'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scores': scores,
        })
        print(f"✓ 视频信息已保存到数据库: {video_id}")
    except Exception as e:
        print(f"❌ 保存到数据库失败: {e}")

    return flask.jsonify({'videoId': video_id})


@app.post('/api/videoinference/list')
def get_video_list():
    """获取视频列表"""
    request_params = flask.request.get_json()

    try:
        number = request_params['pageNumber']
        length = request_params['pageLength']
    except KeyError:
        return flask.abort(400, description="缺少分页参数")

    response_videos = []

    try:
        pagination_videos = database.surveillance.videos.find().limit(length).skip(length * (number - 1))
    except ValueError:
        return flask.abort(400, description="分页参数错误")

    total_count = database.surveillance.videos.count_documents({})

    for video in pagination_videos:
        response_videos.append({
            'name': video['name'],
            'note': video['note'],
            'time': video['time'],
            'videoId': video['videoId'],
        })

    return flask.jsonify({'videos': response_videos, 'totalCount': total_count})


@app.get('/api/videoinference/detail/<string:video_id>')
def get_video_detail(video_id):
    """获取视频详情"""
    video = database.surveillance.videos.find_one({'videoId': video_id})

    if video is None:
        return flask.abort(404, description="视频不存在")

    return flask.jsonify({
        'videoId': video['videoId'],
        'name': video['name'],
        'note': video['note'],
        'time': video['time'],
        'scores': video['scores'],
    })


@app.get('/api/videoinference/cover/<string:video_id>')
def get_video_cover(video_id):
    """获取视频封面"""
    print(f"\n=== 请求封面: {video_id} ===")

    cover_path = f'servers/covers/result.{video_id}.jpg'
    abs_cover_path = os.path.abspath(cover_path)

    print(f"  相对路径: {cover_path}")
    print(f"  绝对路径: {abs_cover_path}")
    print(f"  文件存在: {os.path.exists(abs_cover_path)}")

    if os.path.exists(abs_cover_path):
        file_size = os.path.getsize(abs_cover_path)
        print(f"✓ 找到封面: {abs_cover_path} ({file_size / 1024:.2f} KB)")
    else:
        print(f"❌ 封面不存在: {abs_cover_path}")
        # 列出covers目录下的文件
        covers_dir = os.path.dirname(abs_cover_path)
        if os.path.exists(covers_dir):
            existing_files = os.listdir(covers_dir)
            print(f"  covers目录下的文件: {existing_files}")
        else:
            print(f"  covers目录不存在: {covers_dir}")
        return abort(404, description="封面图片不存在")

    try:
        return send_file(abs_cover_path, mimetype='image/jpeg')
    except Exception as e:
        print(f"❌ 发送封面失败: {e}")
        import traceback
        traceback.print_exc()
        return abort(500, description="封面发送失败")


@app.get('/api/videoinference/video/<string:video_id>')
def get_result_video(video_id):
    """获取结果视频"""
    print(f"\n=== 请求视频: {video_id} ===")
    print(f"  请求来源: {flask.request.remote_addr}")
    print(f"  User-Agent: {flask.request.headers.get('User-Agent', 'Unknown')}")

    possible_paths = [
        f'servers/videos/result.{video_id}.mp4',
        f'servers/videos/result.{video_id}.avi',
    ]

    for filepath in possible_paths:
        abs_filepath = os.path.abspath(filepath)

        if os.path.exists(abs_filepath):
            file_size = os.path.getsize(abs_filepath)
            file_ext = os.path.splitext(filepath)[1]
            print(f"✓ 找到视频: {abs_filepath}")
            print(f"  文件大小: {file_size / (1024 * 1024):.2f} MB")
            print(f"  文件格式: {file_ext}")

            try:
                return send_video_file(filepath)
            except Exception as e:
                print(f"❌ 发送失败: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"❌ 未找到视频: {video_id}")
    print(f"  已检查路径: {possible_paths}")
    return abort(404, description="视频文件不存在")


@app.post('/api/videoinference/delete')
def delete_videos():
    """删除视频"""
    request_params = flask.request.get_json()

    try:
        video_ids = request_params['videoIds']
    except KeyError:
        return flask.abort(400, description="缺少视频ID列表")

    delete_result = database.surveillance.videos.delete_many({'videoId': {'$in': video_ids}})

    for video_id in video_ids:
        remove_queue.put(f'servers/videos/result.{video_id}.mp4')
        remove_queue.put(f'servers/videos/result.{video_id}.avi')
        remove_queue.put(f'servers/covers/result.{video_id}.jpg')

    return flask.jsonify({'deletedCount': delete_result.deleted_count})


@app.post('/api/realtimeinference/create')
def create_realtime_session():
    """创建实时推理会话"""
    request_params = flask.request.get_json()

    try:
        source = request_params['source']
    except KeyError:
        return flask.abort(400, description="缺少视频源")

    try:
        name = request_params['name']
        note = request_params['note']
    except KeyError:
        return flask.abort(400, description="缺少必要参数")

    session_id = str(next(id_generator))

    # 保存到数据库
    database.surveillance.sessions.insert_one({
        'source': source,
        'name': name,
        'note': note,
        'sessionId': session_id,
    })

    # 创建实时推理会话
    try:
        with realtime_sessions_lock:
            realtime_sessions[session_id] = realtime.RealtimeInferenceSession(source)
        print(f"✓ 创建实时会话: {session_id}")
    except Exception as e:
        print(f"❌ 创建实时会话失败: {e}")
        return flask.abort(500, description=f"创建会话失败: {str(e)}")

    return flask.jsonify({'sessionId': session_id})


@app.get('/api/realtimeinference/session/<string:session_id>')
def generate_realtime_frames(session_id):
    """生成实时检测帧流"""
    with realtime_sessions_lock:
        try:
            session = realtime_sessions[session_id]
        except KeyError:
            return flask.abort(404, description="会话不存在")

    return flask.Response(generate_realtime_response(session), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.post('/api/realtimeinference/list')
def get_realtime_sessions():
    """获取实时会话列表"""
    request_params = flask.request.get_json()

    try:
        number = request_params['pageNumber']
        length = request_params['pageLength']
    except KeyError:
        return flask.abort(400, description="缺少分页参数")

    response_sessions = []

    try:
        pagination_sessions = database.surveillance.sessions.find().limit(length).skip(length * (number - 1))
    except ValueError:
        return flask.abort(400, description="分页参数错误")

    total_count = database.surveillance.sessions.count_documents({})

    for session in pagination_sessions:
        response_sessions.append({
            'source': session['source'],
            'name': session['name'],
            'note': session['note'],
            'sessionId': session['sessionId'],
        })

    return flask.jsonify({'sessions': response_sessions, 'totalCount': total_count})


@app.get('/api/realtimeinference/detail/<string:session_id>')
def get_session_detail(session_id):
    """获取会话详情"""
    session = database.surveillance.sessions.find_one({'sessionId': session_id})

    if session is None:
        return flask.abort(404, description="会话不存在")

    return flask.jsonify({
        'name': session['name'],
        'note': session['note'],
        'source': session['source'],
    })


@app.post('/api/realtimeinference/delete')
def delete_realtime_sessions():
    """删除实时会话"""
    request_params = flask.request.get_json()

    try:
        session_ids = request_params['sessionIds']
    except KeyError:
        return flask.abort(400, description="缺少会话ID列表")

    delete_result = database.surveillance.sessions.delete_many({'sessionId': {'$in': session_ids}})

    for session_id in session_ids:
        with realtime_sessions_lock:
            release_and_delete_session(session_id)
            print(f"✓ 删除实时会话: {session_id}")

    return flask.jsonify({'deletedCount': delete_result.deleted_count})


@app.get('/api/realtimeinference/sync')
def sync_realtime_sessions():
    """同步实时会话"""
    sessions = database.surveillance.sessions.find()

    with realtime_sessions_lock:
        # 释放所有现有会话
        for session in realtime_sessions.values():
            session.release()

        realtime_sessions.clear()

        # 重新创建所有会话
        for session in sessions:
            try:
                realtime_sessions[session['sessionId']] = realtime.RealtimeInferenceSession(session['source'])
                print(f"✓ 同步会话: {session['sessionId']}")
            except Exception as e:
                print(f"❌ 同步会话失败 {session['sessionId']}: {e}")

        return flask.jsonify({'sessionCount': len(realtime_sessions)})


