#!/usr/bin/env python3
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave

import actionlib
import rospy
from sound_play.msg import SoundRequest, SoundRequestAction, SoundRequestGoal
from std_msgs.msg import String


class OfflineTtsNode:
    def __init__(self):
        self.subscribe_topic = rospy.get_param("~subscribe_topic", rospy.get_param("/tts/subscribe_topic", "/voice/say"))
        self.legacy_subscribe_topic = rospy.get_param(
            "~legacy_subscribe_topic", rospy.get_param("/tts/legacy_subscribe_topic", "/xfyun/tts")
        )
        self.sound_topic = rospy.get_param("~sound_topic", rospy.get_param("/tts/sound_topic", "/robotsound"))
        self.wav_path = rospy.get_param("~wav_path", rospy.get_param("/tts/wav_path", "/dev/shm/offline_tts.wav"))
        self.piper_executable = rospy.get_param(
            "~piper_executable", rospy.get_param("/tts/piper_executable", "piper")
        )
        self.engine = rospy.get_param("~engine", rospy.get_param("/tts/engine", "auto"))
        self.flite_executable = rospy.get_param("~flite_executable", rospy.get_param("/tts/flite_executable", "flite"))
        self.flite_voice = rospy.get_param("~flite_voice", rospy.get_param("/tts/flite_voice", "slt"))
        self.model_path = rospy.get_param("~model_path", rospy.get_param("/tts/model_path", ""))
        self.config_path = rospy.get_param("~config_path", rospy.get_param("/tts/config_path", ""))
        self.volume = float(rospy.get_param("~volume", rospy.get_param("/tts/volume", 1.0)))
        self.length_scale = float(rospy.get_param("~length_scale", rospy.get_param("/tts/length_scale", 1.0)))
        self.noise_scale = float(rospy.get_param("~noise_scale", rospy.get_param("/tts/noise_scale", 0.667)))
        self.noise_w = float(rospy.get_param("~noise_w", rospy.get_param("/tts/noise_w", 0.8)))
        self.sound_wait_timeout = float(rospy.get_param("~sound_wait_timeout", rospy.get_param("/tts/sound_wait_timeout", 10.0)))
        self.sound_action = rospy.get_param("~sound_action", rospy.get_param("/tts/sound_action", "sound_play"))
        self.playback_backend = str(
            rospy.get_param("~playback_backend", rospy.get_param("/tts/playback_backend", "sound_play"))
        ).strip().lower()
        self.audio_device = str(rospy.get_param("~audio_device", rospy.get_param("/tts/audio_device", ""))).strip()
        self.audio_retry_seconds = float(
            rospy.get_param("~audio_retry_seconds", rospy.get_param("/tts/audio_retry_seconds", 5.0))
        )
        self.audio_retry_interval = float(
            rospy.get_param("~audio_retry_interval", rospy.get_param("/tts/audio_retry_interval", 0.2))
        )
        self.blocking_playback = bool(rospy.get_param("~blocking_playback", rospy.get_param("/tts/blocking_playback", True)))
        self.playback_margin = float(rospy.get_param("~playback_margin", rospy.get_param("/tts/playback_margin", 0.6)))
        self.wav_keep_count = max(1, int(rospy.get_param("~wav_keep_count", rospy.get_param("/tts/wav_keep_count", 8))))
        self._sequence = 0
        self._generated_wavs = []
        self._lock = threading.Lock()

        self.play_pub = rospy.Publisher(self.sound_topic, SoundRequest, queue_size=10)
        self.action_client = actionlib.SimpleActionClient(self.sound_action, SoundRequestAction)
        rospy.Subscriber(self.subscribe_topic, String, self._speak_callback, queue_size=10)
        if self.legacy_subscribe_topic and self.legacy_subscribe_topic != self.subscribe_topic:
            rospy.Subscriber(self.legacy_subscribe_topic, String, self._speak_callback, queue_size=10)

        self._validate_runtime()
        rospy.loginfo(
            "offline_tts_node ready: %s and %s -> %s (backend=%s, device=%s)",
            self.subscribe_topic,
            self.legacy_subscribe_topic,
            self.sound_topic,
            self.playback_backend,
            self.audio_device or "default",
        )

    def _validate_runtime(self):
        if self._piper_available():
            rospy.loginfo("Using Piper TTS engine")
            return
        if self.engine == "piper":
            rospy.logwarn("Piper executable not found: %s. Run tools/setup_offline_voice.sh first.", self.piper_executable)
            return
        if shutil.which(self.flite_executable):
            rospy.logwarn("Piper not ready; falling back to flite for offline smoke tests")
            return
        rospy.logwarn("No offline TTS engine is ready. Piper=%s flite=%s", self.piper_executable, self.flite_executable)

    def _piper_available(self):
        executable = shutil.which(self.piper_executable) if os.path.sep not in self.piper_executable else self.piper_executable
        if not executable or not os.path.exists(executable):
            return False
        if not self.model_path or not os.path.exists(self.model_path):
            return False
        if self.config_path and not os.path.exists(self.config_path):
            return False
        return True

    def _speak_callback(self, msg):
        text = msg.data.strip()
        if not text:
            return
        with self._lock:
            try:
                wav_path = self._next_wav_path()
                self._synthesize(text, wav_path)
                self._play_wav(wav_path)
                self._remember_wav(wav_path)
            except Exception as exc:
                rospy.logerr("Offline TTS failed: %s", exc)

    def _next_wav_path(self):
        wav_dir = os.path.dirname(self.wav_path) or "."
        os.makedirs(wav_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.wav_path))[0] or "offline_tts"
        self._sequence += 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(wav_dir, "%s_%s_%03d.wav" % (base, stamp, self._sequence))

    def _remember_wav(self, wav_path):
        self._generated_wavs.append(wav_path)
        while len(self._generated_wavs) > self.wav_keep_count:
            old_path = self._generated_wavs.pop(0)
            if old_path != wav_path and os.path.exists(old_path):
                try:
                    os.unlink(old_path)
                except OSError:
                    pass

    @staticmethod
    def _wav_duration(wav_path):
        try:
            with wave.open(wav_path, "rb") as wav_file:
                rate = float(wav_file.getframerate())
                if rate <= 0:
                    return 0.0
                return wav_file.getnframes() / rate
        except Exception:
            return 0.0

    def _synthesize(self, text, wav_path):
        if self.engine in ("auto", "piper") and self._piper_available():
            self._synthesize_with_piper(text, wav_path)
            return
        if self.engine == "piper":
            raise RuntimeError("piper is selected but executable/model/config is missing")
        if shutil.which(self.flite_executable):
            self._synthesize_with_flite(text, wav_path)
            return
        raise RuntimeError("no usable offline TTS engine found")

    def _synthesize_with_piper(self, text, wav_path):
        if not shutil.which(self.piper_executable) and not os.path.exists(self.piper_executable):
            raise RuntimeError("piper executable is missing")
        if not os.path.exists(self.model_path):
            raise RuntimeError("piper model is missing: %s" % self.model_path)

        wav_dir = os.path.dirname(wav_path) or "."
        os.makedirs(wav_dir, exist_ok=True)
        fd, tmp_wav = tempfile.mkstemp(prefix="offline_tts_", suffix=".wav", dir=wav_dir)
        os.close(fd)

        cmd = [
            self.piper_executable,
            "--model",
            self.model_path,
            "--output_file",
            tmp_wav,
            "--length_scale",
            str(self.length_scale),
            "--noise_scale",
            str(self.noise_scale),
            "--noise_w",
            str(self.noise_w),
        ]
        if self.config_path and os.path.exists(self.config_path):
            cmd.extend(["--config", self.config_path])

        try:
            subprocess.run(cmd, input=text, text=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.replace(tmp_wav, wav_path)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            raise RuntimeError("piper exited with %s: %s" % (exc.returncode, stderr))
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)

    def _synthesize_with_flite(self, text, wav_path):
        wav_dir = os.path.dirname(wav_path) or "."
        os.makedirs(wav_dir, exist_ok=True)
        fd, tmp_wav = tempfile.mkstemp(prefix="offline_tts_", suffix=".wav", dir=wav_dir)
        os.close(fd)
        cmd = [self.flite_executable, "-voice", self.flite_voice, "-t", text, "-o", tmp_wav]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.replace(tmp_wav, wav_path)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip() if exc.stderr else ""
            raise RuntimeError("flite exited with %s: %s" % (exc.returncode, stderr))
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)

    def _make_sound_request(self, wav_path):
        req = SoundRequest()
        req.sound = SoundRequest.PLAY_FILE
        req.command = SoundRequest.PLAY_ONCE
        req.volume = self.volume
        req.arg = wav_path
        return req

    def _play_wav_with_aplay(self, wav_path, duration):
        """Play directly through ALSA; do not depend on sound_play/GStreamer state."""
        cmd = ["aplay", "-q"]
        if self.audio_device:
            cmd.extend(["-D", self.audio_device])
        cmd.append(wav_path)
        deadline = time.monotonic() + max(0.0, self.audio_retry_seconds)
        while not rospy.is_shutdown():
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                rospy.loginfo("Spoken: %s (%.2fs, ALSA direct)", wav_path, duration)
                return
            except FileNotFoundError:
                raise RuntimeError("aplay is not available in this runtime")
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode("utf-8", errors="replace").strip() if exc.stderr else ""
                if "Device or resource busy" not in stderr or time.monotonic() >= deadline:
                    raise RuntimeError("aplay exited with %s: %s" % (exc.returncode, stderr))
                rospy.logwarn_throttle(1.0, "Speaker busy; retrying direct ALSA playback")
                time.sleep(max(0.05, self.audio_retry_interval))

    def _play_wav(self, wav_path):
        duration = self._wav_duration(wav_path)
        if self.playback_backend in ("aplay", "alsa", "alsa_direct"):
            self._play_wav_with_aplay(wav_path, duration)
            return

        req = self._make_sound_request(wav_path)
        wait_duration = max(0.6, duration + self.playback_margin)

        if self.blocking_playback:
            if self.action_client.wait_for_server(rospy.Duration(self.sound_wait_timeout)):
                goal = SoundRequestGoal()
                goal.sound_request = req
                self.action_client.send_goal(goal)
                if self.action_client.wait_for_result(rospy.Duration(wait_duration + self.sound_wait_timeout)):
                    rospy.loginfo("Spoken: %s (%.2fs, blocking)", wav_path, duration)
                    return
                self.action_client.cancel_goal()
                rospy.logwarn("Timed out waiting for sound_play action; falling back to topic playback")
            else:
                rospy.logwarn("sound_play action [%s] is unavailable; falling back to topic playback", self.sound_action)

        deadline = time.time() + self.sound_wait_timeout
        while self.play_pub.get_num_connections() == 0 and not rospy.is_shutdown() and time.time() < deadline:
            time.sleep(0.05)
        self.play_pub.publish(req)
        time.sleep(wait_duration)
        rospy.loginfo("Spoken: %s (%.2fs, topic fallback)", wav_path, duration)


if __name__ == "__main__":
    rospy.init_node("offline_tts_node")
    OfflineTtsNode()
    rospy.spin()
