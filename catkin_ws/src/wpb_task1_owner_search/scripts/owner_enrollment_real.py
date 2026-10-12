#!/usr/bin/env python3
"""Safe standalone multi-angle owner-face enrollment for the WPB robot."""

from __future__ import print_function

import datetime
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


def normalize(value):
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    magnitude = float(np.linalg.norm(vector))
    return vector / magnitude if magnitude > 1e-8 else None


def similarity(left, right):
    left, right = normalize(left), normalize(right)
    return float(np.dot(left, right)) if left is not None and right is not None else -1.0


def timestamp():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def slug(value):
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff·_-]+", "_", value).strip("_")[:32] or "owner"


class OwnerEnrollment(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.image_lock = threading.Lock()
        self.latest_image = None
        self.latest_stamp = None
        self.asr_lock = threading.Lock()
        self.asr_messages = []

        self.image_topic = rospy.get_param("~image_topic", "/kinect2/qhd/image_color_rect")
        self.say_topic = rospy.get_param("~say_topic", "/voice/say")
        self.asr_topic = rospy.get_param("~asr_topic", "/voice/asr_text")
        self.owner_store_dir = os.path.expanduser(rospy.get_param("~owner_store_dir"))
        self.session_dir = os.path.expanduser(rospy.get_param("~session_dir"))
        self.max_owners = max(1, int(rospy.get_param("~max_owners", 3)))
        self.capture_timeout = float(rospy.get_param("~capture_timeout", 20.0))
        self.settle_seconds = float(rospy.get_param("~capture_settle_seconds", 2.0))
        self.capture_angles = rospy.get_param("~capture_angles")
        self.min_face_pixels = int(rospy.get_param("~min_face_pixels", 100))
        self.min_face_score = float(rospy.get_param("~min_face_det_score", 0.55))
        self.min_sharpness = float(rospy.get_param("~min_face_sharpness", 55.0))
        self.min_consistency = float(rospy.get_param("~min_owner_face_consistency", 0.42))
        self.duplicate_similarity = float(rospy.get_param("~duplicate_similarity", 0.55))
        self.name_attempts = max(1, int(rospy.get_param("~name_confirmation_attempts", 2)))
        self.asr_wait = float(rospy.get_param("~asr_wait_seconds", 15.0))
        self.asr_post_tts_silence = float(rospy.get_param("~asr_post_tts_silence_seconds", 4.5))
        self.qwen_url = rospy.get_param("~qwen_url", "http://127.0.0.1:11434/api/chat")
        self.qwen_model = rospy.get_param("~qwen_model", "qwen3.5:0.8b")
        self.qwen_timeout = float(rospy.get_param("~qwen_timeout_seconds", 12.0))
        self.qwen_warmup_timeout = float(rospy.get_param("~qwen_warmup_timeout_seconds", 60.0))
        self.qwen_require_gpu = bool(rospy.get_param("~qwen_require_gpu", True))
        self.face_model_root = os.path.expanduser(rospy.get_param("~face_model_root", "~/.insightface"))
        self.face_model_name = rospy.get_param("~face_model_name", "buffalo_sc")
        self.face_ctx_id = int(rospy.get_param("~face_ctx_id", -1))
        self.face_det_size = int(rospy.get_param("~face_det_size", 480))
        self.face_det_thresh = float(rospy.get_param("~face_det_thresh", 0.35))

        self.say_pub = rospy.Publisher(self.say_topic, String, queue_size=10)
        rospy.Subscriber(self.image_topic, Image, self.on_image, queue_size=1)
        rospy.Subscriber(self.asr_topic, String, self.on_asr, queue_size=20)
        self.face_app = self.init_face_model()
        if self.face_app is None:
            raise RuntimeError("InsightFace 模型不可用，程序已安全退出。")
        os.makedirs(self.owner_store_dir, exist_ok=True)
        os.makedirs(self.session_dir, exist_ok=True)

    def init_face_model(self):
        try:
            from insightface.app import FaceAnalysis
            cache = os.path.join(self.face_model_root, "models", self.face_model_name)
            if not os.path.isdir(cache) or not any(x.endswith(".onnx") for x in os.listdir(cache)):
                rospy.logerr("InsightFace cached model missing: %s", cache)
                return None
            providers = ["CPUExecutionProvider"] if self.face_ctx_id < 0 else None
            app = FaceAnalysis(name=self.face_model_name, root=self.face_model_root, providers=providers)
            app.prepare(ctx_id=self.face_ctx_id, det_size=(self.face_det_size, self.face_det_size),
                        det_thresh=self.face_det_thresh)
            return app
        except Exception as exc:
            rospy.logerr("InsightFace startup failed: %s", exc)
            return None

    def on_image(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, "bgr8")
            with self.image_lock:
                self.latest_image = image.copy()
                self.latest_stamp = message.header.stamp.to_sec() or time.time()
        except Exception as exc:
            rospy.logwarn_throttle(3.0, "Kinect image conversion failed: %s", exc)

    def on_asr(self, message):
        text = message.data.strip()
        if text:
            with self.asr_lock:
                self.asr_messages.append((time.time(), text))
                self.asr_messages = self.asr_messages[-30:]

    def say(self, text, hold=1.0):
        rospy.loginfo("Enrollment TTS: %s", text)
        self.say_pub.publish(String(data=text))
        rospy.sleep(max(0.0, hold))

    def image(self, timeout=2.0):
        deadline = time.time() + timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.image_lock:
                if self.latest_image is not None:
                    return self.latest_image.copy(), self.latest_stamp
            rospy.sleep(0.1)
        return None, None

    def inspect_face(self, image):
        candidates = []
        for face in self.face_app.get(image):
            box = np.asarray(face.bbox, dtype=np.int32)
            width, height = int(box[2] - box[0]), int(box[3] - box[1])
            score = float(getattr(face, "det_score", 0.0))
            if width >= self.min_face_pixels and height >= self.min_face_pixels and score >= self.min_face_score:
                candidates.append((face, box, score))
        if len(candidates) != 1:
            return None, "请确保画面内只有一张清晰的人脸" if candidates else "未检测到清晰人脸"
        face, box, score = candidates[0]
        x1, y1, x2, y2 = [max(0, int(x)) for x in box]
        crop = image[y1:y2, x1:x2]
        sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if crop.size else 0.0
        embedding = normalize(getattr(face, "normed_embedding", []))
        if sharpness < self.min_sharpness or embedding is None:
            return None, "照片偏模糊，请正视摄像头并保持不动"
        return {"box": [x1, y1, x2, y2], "score": score, "sharpness": sharpness, "embedding": embedding}, None

    def capture(self, owner_number, angle_number, instruction, staging):
        self.say("第%d位主人，请%s。" % (owner_number, instruction), self.settle_seconds)
        deadline = time.time() + self.capture_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            image, ros_stamp = self.image()
            if image is None:
                continue
            face, reason = self.inspect_face(image)
            if face is None:
                rospy.logwarn_throttle(2.0, "Enrollment capture: %s", reason)
                rospy.sleep(0.25)
                continue
            filename = "angle_%02d.jpg" % (angle_number + 1)
            cv2.imwrite(os.path.join(staging, filename), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            proof = image.copy()
            x1, y1, x2, y2 = face["box"]
            cv2.rectangle(proof, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(proof, "pending %s" % timestamp(), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
            cv2.imwrite(os.path.join(staging, "angle_%02d_evidence.jpg" % (angle_number + 1)), proof)
            face.update({"file": filename, "instruction": instruction, "captured_at": timestamp(), "image_stamp": ros_stamp})
            self.say("已记录。", 0.4)
            return face
        raise RuntimeError("拍摄超时：%s" % instruction)

    def hear(self, prompt):
        with self.asr_lock:
            self.asr_messages = []
        self.say(prompt, 0.8)
        # faster-whisper emits 4-second chunks. Discard a whole chunk after
        # TTS so the robot's own voice and room echo cannot become a reply.
        rospy.sleep(max(0.0, self.asr_post_tts_silence))
        with self.asr_lock:
            started = time.time()
            self.asr_messages = [(when, text) for when, text in self.asr_messages if when >= started]
        deadline = started + self.asr_wait
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.asr_lock:
                heard = [text for when, text in self.asr_messages if when >= started]
            if heard:
                return heard[-1]
            rospy.sleep(0.1)
        return ""

    def qwen(self, system, user):
        payload = {"model": self.qwen_model, "stream": False, "format": "json", "think": False,
                   "options": {"temperature": 0, "num_predict": 64},
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        request = urllib.request.Request(self.qwen_url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.qwen_timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            return json.loads(result.get("message", {}).get("content", "{}"))
        except Exception as exc:
            rospy.logwarn("Qwen structured dialog unavailable: %s", exc)
            return {}

    def warmup_qwen(self):
        """Load Qwen before enrollment and require a GPU-backed runner."""
        payload = {"model": self.qwen_model, "stream": False, "format": "json", "think": False,
                   "options": {"temperature": 0, "num_predict": 8},
                   "messages": [{"role": "system", "content": "只输出空 JSON 对象。"},
                                {"role": "user", "content": "预热"}]}
        request = urllib.request.Request(self.qwen_url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.qwen_warmup_timeout) as response:
                if response.getcode() != 200:
                    raise RuntimeError("Qwen returned HTTP %s" % response.getcode())
            ps_url = self.qwen_url.rsplit("/api/chat", 1)[0] + "/api/ps"
            with urllib.request.urlopen(ps_url, timeout=5) as response:
                models = json.loads(response.read().decode("utf-8")).get("models", [])
            active = [item for item in models if item.get("name") == self.qwen_model]
            vram = max([int(item.get("size_vram", 0)) for item in active] or [0])
            if self.qwen_require_gpu and vram <= 0:
                raise RuntimeError("Qwen 未使用 GPU，拒绝开始录入")
            rospy.loginfo("Qwen warmup ready: model=%s size_vram=%d", self.qwen_model, vram)
        except Exception as exc:
            raise RuntimeError("Qwen 预热失败：%s" % exc)

    def name_from(self, speech):
        result = self.qwen("你是中文姓名提取器。只输出 JSON：{\"name\": 字符串或null}。仅在话语明确说出本人姓名时返回姓名，不要猜测。", speech)
        name = str(result.get("name") or "").strip()
        if not name:
            match = re.search(r"(?:我叫|我是|名字是|姓名是)\s*([\u4e00-\u9fffA-Za-z·]{2,16})", speech)
            name = match.group(1) if match else ""
        return name if re.match(r"^[\u4e00-\u9fffA-Za-z·]{2,16}$", name) else ""

    def yes_no(self, speech):
        value = self.qwen("判断中文回答是否明确同意。只输出 JSON：{\"confirmed\": true、false 或null}。不明确为null。", speech).get("confirmed")
        if isinstance(value, bool):
            return value
        if re.search(r"不是|不对|不要|否|错", speech):
            return False
        return True if re.search(r"是的|是|对|没错|确认|可以", speech) else None

    def get_confirmed_name(self):
        for unused in range(self.name_attempts):
            spoken = self.hear("请清楚地说：我叫某某某。")
            name = self.name_from(spoken)
            if not name:
                self.say("没有听清姓名，我们再说一次。", 0.8)
                continue
            confirmation = self.hear("我记录您是%s。请回答是或不是。" % name)
            if self.yes_no(confirmation) is True:
                return name, spoken, confirmation
            self.say("姓名未确认，我们重新录入。", 0.8)
        raise RuntimeError("姓名未得到本人明确确认")

    def existing(self):
        for dirname in os.listdir(self.owner_store_dir):
            path = os.path.join(self.owner_store_dir, dirname, "manifest.json")
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                embedding = normalize(manifest.get("mean_embedding", []))
                if embedding is not None:
                    yield manifest, embedding
            except (OSError, ValueError):
                pass

    def commit(self, owner_number, name, captures, staging, spoken, confirmation):
        vectors = np.asarray([capture["embedding"] for capture in captures], dtype=np.float32)
        mean = normalize(np.mean(vectors, axis=0))
        quality = [similarity(vector, mean) for vector in vectors]
        if min(quality) < self.min_consistency:
            raise RuntimeError("多角度人脸特征不一致，可能换人；未保存。")
        for old, vector in self.existing():
            score = similarity(mean, vector)
            if score >= self.duplicate_similarity:
                raise RuntimeError("疑似与已登记主人%s重复（相似度%.2f）；为避免错绑，未保存。" %
                                   (old.get("display_name", "未知"), score))
        owner_id = "owner_%02d_%s_%s" % (owner_number, slug(name), datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        manifest = {"schema_version": 1, "owner_id": owner_id, "display_name": name, "enrolled_at": timestamp(),
                    "identity_confirmed_by": "spoken_name_yes_no", "name_transcript": spoken,
                    "confirmation_transcript": confirmation, "face_model": self.face_model_name,
                    "mean_embedding": mean.astype(float).tolist(), "minimum_intra_owner_similarity": min(quality),
                    "captures": []}
        for capture, score in zip(captures, quality):
            with open(os.path.join(staging, capture["file"]), "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            manifest["captures"].append({"file": capture["file"], "evidence_file": capture["file"].replace(".jpg", "_evidence.jpg"),
                "instruction": capture["instruction"], "captured_at": capture["captured_at"], "det_score": capture["score"],
                "sharpness": capture["sharpness"], "intra_owner_similarity": score, "sha256": digest})
        with open(os.path.join(staging, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        final_dir = os.path.join(self.owner_store_dir, owner_id)
        os.rename(staging, final_dir)
        return final_dir

    def enroll_one(self, owner_number):
        staging = tempfile.mkdtemp(prefix="owner_%02d_" % owner_number, dir=self.session_dir)
        try:
            captures = [self.capture(owner_number, number, instruction, staging)
                        for number, instruction in enumerate(self.capture_angles)]
            name, spoken, confirmation = self.get_confirmed_name()
            path = self.commit(owner_number, name, captures, staging, spoken, confirmation)
            self.say("%s，录入完成。" % name, 1.0)
            rospy.loginfo("Owner enrollment saved: %s", path)
            return True
        except Exception as exc:
            rospy.logerr("Owner %d not saved: %s", owner_number, exc)
            self.say("本次录入未保存。%s" % str(exc), 1.0)
            shutil.rmtree(staging, ignore_errors=True)
            return False

    def run(self):
        self.say("正在预热姓名识别模型。", 0.2)
        self.warmup_qwen()
        self.say("主人录入程序已启动。机器人不会移动。请每次只有一位主人站在正前方。", 1.5)
        if self.image(15.0)[0] is None:
            raise RuntimeError("未收到 Kinect 彩色图像")
        completed = 0
        for owner_number in range(1, self.max_owners + 1):
            self.say("现在开始录入第%d位主人。" % owner_number, 1.0)
            completed += int(self.enroll_one(owner_number))
        self.say("主人录入结束，成功保存%d位主人。" % completed, 1.0)


if __name__ == "__main__":
    rospy.init_node("owner_enrollment_real")
    OwnerEnrollment().run()
