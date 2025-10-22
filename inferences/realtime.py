import threading
import collections
import cv2
import time
import toml
import numpy as np
import inferences.engines as engines

configs = toml.load('inferences/configs/config.toml')

segment_length = configs['segment-length']
history_length = configs.get('history-length', 10)  # 默认值10

capture_interval = configs.get('capture-interval', 0.03)  # 默认30fps
prepare_interval = configs.get('prepare-interval', 0.03)
predict_interval = configs.get('predict-interval', 0.5)


def execute_task_in_seconds(task, args=None, target_seconds=0):
    start_seconds = time.perf_counter()

    if args is None:
        task_execution_result = task()
    else:
        task_execution_result = task(args)

    finish_seconds = time.perf_counter()

    delta_seconds = finish_seconds - start_seconds
    delay_seconds = target_seconds - delta_seconds

    if delay_seconds > 0:
        time.sleep(delay_seconds)

    return task_execution_result


class RealtimeInferenceSession:
    def __init__(self, source):
        self.capture = cv2.VideoCapture(source)

        self.segment_queue = collections.deque(maxlen=segment_length)
        self.feature_queue = collections.deque(maxlen=history_length)

        self.capture_running = True
        self.prepare_running = True
        self.predict_running = True

        self.current_frame = None
        self.current_score = 0.0  # 初始化为0.0而不是None

        self.current_lock = threading.Lock()
        self.segment_lock = threading.Lock()

        self.capture_thread = threading.Thread(target=self.capture_process, daemon=True)
        self.prepare_thread = threading.Thread(target=self.prepare_process, daemon=True)
        self.predict_thread = threading.Thread(target=self.predict_process, daemon=True)

        self.capture_thread.start()
        self.prepare_thread.start()
        self.predict_thread.start()

    def __del__(self):
        self.release()

    def capture_task(self):
        read_success, captured_frame = self.capture.read()

        if read_success:
            with self.current_lock:
                self.current_frame = captured_frame

    def capture_process(self):
        while self.capture_running:
            execute_task_in_seconds(self.capture_task, target_seconds=capture_interval)

    def prepare_task(self):
        with self.current_lock:
            current_frame = self.current_frame

        if current_frame is not None:
            with self.segment_lock:
                # 使用 engines.frame_preprocess 预处理帧
                self.segment_queue.append(engines.frame_preprocess(current_frame))

    def prepare_process(self):
        while self.prepare_running:
            execute_task_in_seconds(self.prepare_task, target_seconds=prepare_interval)

    def load_segment_frames(self):
        if not len(self.segment_queue) == segment_length:
            current_segment_frames = None
        else:
            current_segment_frames = list(self.segment_queue)
            self.segment_queue.clear()

        return current_segment_frames

    def predict_task(self):
        with self.segment_lock:
            segment_frames = self.load_segment_frames()

        if segment_frames is not None:
            try:
                # 提取当前段的特征
                segment_features = engines.extract_segment_features(segment_frames)

                # 添加到特征队列
                self.feature_queue.append(segment_features)

                # 如果特征队列有足够的历史数据，进行检测
                if len(self.feature_queue) > 0:
                    # 堆叠所有历史特征
                    all_features = np.concatenate(list(self.feature_queue), axis=0)

                    # 进行异常检测
                    realtime_scores = engines.detection_by_features(all_features)

                    # 更新当前分数（取最后一个分数）
                    with self.current_lock:
                        self.current_score = float(realtime_scores[-1])

            except Exception as e:
                print(f"预测任务出错: {e}")
                with self.current_lock:
                    self.current_score = 0.0

    def predict_process(self):
        while self.predict_running:
            execute_task_in_seconds(self.predict_task, target_seconds=predict_interval)

    def get_result(self):
        with self.current_lock:
            result_frame = self.current_frame
            result_score = self.current_score

        if result_frame is None:
            return None

        # 在帧上绘制检测结果
        return engines.draw_detection_result(result_frame.copy(), result_score)

    def release(self):
        self.capture_running = False
        self.prepare_running = False
        self.predict_running = False

        # 等待线程结束（最多等待1秒）
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.prepare_thread.is_alive():
            self.prepare_thread.join(timeout=1.0)
        if self.predict_thread.is_alive():
            self.predict_thread.join(timeout=1.0)

        if self.capture.isOpened():
            self.capture.release()