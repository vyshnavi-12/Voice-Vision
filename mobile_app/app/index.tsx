import React, { useEffect, useRef } from 'react';
import { View, StyleSheet } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Location from 'expo-location';
import * as FileSystem from 'expo-file-system/legacy';
import { Audio } from 'expo-av';

const BACKEND_URL      = "http://172.20.10.5:5000";
const AWAKE_TIMEOUT_MS = 10000;

const RECORDING_OPTIONS: Audio.RecordingOptions = {
  android: {
    extension: '.m4a',
    outputFormat: 2,
    audioEncoder: 3,
    sampleRate: 16000,
    numberOfChannels: 1,
    bitRate: 128000,
  },
  ios: {
    extension: '.wav',
    audioQuality: 0x7F,
    sampleRate: 16000,
    numberOfChannels: 1,
    bitRate: 128000,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  },
  web: {},
};

export default function HomeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef       = useRef<CameraView>(null);
  const alertSound      = useRef(new Audio.Sound());
  const isSystemBusyRef = useRef(false);
  const isAwakeRef      = useRef(false);
  const loopRunningRef  = useRef(false);
  const awakeTimerRef   = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isProcessingRef = useRef(false);

  useEffect(() => {
    (async () => {
      await requestPermission();
      await Audio.requestPermissionsAsync();
      await Location.requestForegroundPermissionsAsync();

      await Audio.setAudioModeAsync({
        allowsRecordingIOS:      true,
        playsInSilentModeIOS:    true,
        staysActiveInBackground: true,
        shouldDuckAndroid:       false,
      });

      try {
        await alertSound.current.loadAsync(require('../assets/alert.wav'));
        await alertSound.current.setVolumeAsync(1.0);
      } catch (e) { console.log("Alert sound load error:", e); }

      const obstacleInterval = setInterval(backgroundObstacleCheck, 2000);
      runVoiceLoop();

      return () => {
        clearInterval(obstacleInterval);
        loopRunningRef.current = false;
        clearAwakeTimer();
      };
    })();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Helpers ───────────────────────────────────────────────────────────────

  const clearAwakeTimer = () => {
    if (awakeTimerRef.current) {
      clearTimeout(awakeTimerRef.current);
      awakeTimerRef.current = null;
    }
  };

  const resetAwakeTimer = () => {
    clearAwakeTimer();
    awakeTimerRef.current = setTimeout(async () => {
      if (isProcessingRef.current) {
        resetAwakeTimer(); // still processing — reschedule
        return;
      }
      await goToSleep("10s inactivity timeout");
    }, AWAKE_TIMEOUT_MS);
  };

  const speakText = async (text: string, lang = 'en') => {
    try {
      const res = await fetch(`${BACKEND_URL}/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, lang }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.audio) await playAudioFromB64(data.audio);
      }
    } catch (e) { console.log("speakText error:", e); }
  };

  const goToSleep = async (reason: string) => {
    if (!isAwakeRef.current) return;
    console.log(`Going to sleep — reason: ${reason}`);
    isAwakeRef.current = false;
    clearAwakeTimer();
    await speakText("Going to sleep.");
    isSystemBusyRef.current = false;
    console.log("Obstacle detection resumed.");
  };

  // ── Obstacle detection ────────────────────────────────────────────────────
  const backgroundObstacleCheck = async () => {
    if (isSystemBusyRef.current || !cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({ base64: true, quality: 0.1 });
      if (isSystemBusyRef.current) return;
      const res = await fetch(`${BACKEND_URL}/check_obstacle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: photo?.base64 }),
      });
      const data = await res.json();
      if (data.alert && !isSystemBusyRef.current) {
        try { await alertSound.current.replayAsync(); } catch {}
      }
    } catch {}
  };


  // ── OCR guidance loop ─────────────────────────────────────────────────────
  // Called automatically when backend says camera is not aligned.
  // Keeps sending frames every 2.5s until:
  //   - Camera is aligned → OCR runs → result spoken → goToSleep
  //   - 30s total timeout → goToSleep
  // The user NEVER needs to speak again — just adjust the camera.
  const runOcrGuidanceLoop = async () => {
    const MAX_ATTEMPTS  = 12;  // 12 x 2.5s = 30s max
    const RETRY_DELAY_MS = 2500;

    for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
      console.log(`OCR guidance loop — attempt ${attempt + 1}/${MAX_ATTEMPTS}`);

      // Wait for user to adjust camera
      await new Promise(r => setTimeout(r, RETRY_DELAY_MS));

      // Take fresh photo
      let photoBase64: string | null = null;
      if (cameraRef.current) {
        try {
          const photo = await cameraRef.current.takePictureAsync({ base64: true, quality: 0.3 });
          photoBase64 = photo?.base64 ?? null;
        } catch {}
      }

      if (!photoBase64) continue;

      try {
        const res = await fetch(`${BACKEND_URL}/check_ocr`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: photoBase64, lang: 'en' }),
        });

        if (!res.ok) { console.log("check_ocr error:", res.status); continue; }

        const data = await res.json();
        console.log(`OCR loop result: guidance=${data.ocr_needs_guidance} | response=${data.response?.slice(0, 60)}`);

        // Always play the audio (guidance or final result)
        if (data.audio) await playAudioFromB64(data.audio);

        if (!data.ocr_needs_guidance) {
          // Camera is aligned and OCR is done — go to sleep
          await goToSleep("OCR completed");
          return;
        }
        // Still needs guidance — loop continues, user keeps adjusting

      } catch (e) {
        console.log("OCR guidance loop error:", e);
      }
    }

    // Timeout — user couldn't align camera in 30s
    await speakText("Could not read the text. Please try again.");
    await goToSleep("OCR guidance timeout");
  };

  // ── Voice loop ────────────────────────────────────────────────────────────
  const runVoiceLoop = async () => {
    if (loopRunningRef.current) return;
    loopRunningRef.current = true;

    while (loopRunningRef.current) {
      try {
        const { recording } = await Audio.Recording.createAsync(RECORDING_OPTIONS);
        await new Promise(r => setTimeout(r, 3500));
        await recording.stopAndUnloadAsync();
        const uri = recording.getURI();
        if (uri) await sendToBackend(uri, isAwakeRef.current);
        await new Promise(r => setTimeout(r, 150));
      } catch (err) {
        console.log("Voice loop error:", err);
        await goToSleep("voice loop error");
        await new Promise(r => setTimeout(r, 2000));
      }
    }
  };

  // ── Send to backend ───────────────────────────────────────────────────────
  const sendToBackend = async (uri: string, isAwake: boolean) => {
    if (isAwake) isProcessingRef.current = true;
    try {
      let photoBase64: string | null = null;
      if (cameraRef.current) {
        try {
          const photo = await cameraRef.current.takePictureAsync({ base64: true, quality: 0.3 });
          photoBase64 = photo?.base64 ?? null;
        } catch {}
      }

      const audioBase64 = await FileSystem.readAsStringAsync(uri, { encoding: 'base64' });

      const res = await fetch(`${BACKEND_URL}/process_command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audio: audioBase64, image: photoBase64, is_awake: isAwake }),
      });

      if (res.status === 204) return; // IDLE — no speech
      if (!res.ok) { console.log("Backend error:", res.status); return; }

      const data = await res.json();
      console.log("Intent:", data.intent, "| needs_more_info:", data.needs_more_info, "| ocr_guidance:", data.ocr_needs_guidance);

      // ── Wake word detected ──────────────────────────────────────────────
      if (data.intent === 'WAKE_WORD_DETECTED') {
        isAwakeRef.current      = true;
        isSystemBusyRef.current = true;
        if (data.audio) await playAudioFromB64(data.audio);
        resetAwakeTimer();
        console.log("Awake — obstacles paused, 10s timer started.");
        return;
      }

      if (!isAwake) return;

      // Play response audio
      if (data.audio) await playAudioFromB64(data.audio);

      if (data.intent === 'STOP') {
        // User said stop
        await goToSleep("user said STOP");

      } else if (data.ocr_needs_guidance) {
        // OCR guidance: camera not aligned — automatically keep checking
        // until aligned. User just adjusts the camera, no speech needed.
        console.log("OCR guidance — starting auto frame loop...");
        await runOcrGuidanceLoop();
        // runOcrGuidanceLoop handles sleep internally when done or timed out

      } else if (data.needs_more_info) {
        // Follow-up question (e.g. "What is the person's name?")
        // Timer already restarted inside playAudioFromB64 after audio finishes.
        // Just log — no need to call resetAwakeTimer here.
        console.log("Follow-up question — user has fresh 10s after audio ends.");

      } else if (data.intent && data.intent !== 'UNKNOWN' && data.intent !== 'IDLE') {
        // Command fully handled — go to sleep
        await goToSleep("command completed");

      }
      // UNKNOWN — stay awake, let timer count down naturally

      // Emergency GPS
      if (data.intent === 'EMERGENCY_REQUESTED') {
        try {
          const loc = await Location.getCurrentPositionAsync({});
          await fetch(`${BACKEND_URL}/trigger_emergency`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              location: { latitude: loc.coords.latitude, longitude: loc.coords.longitude },
            }),
          });
        } catch (e) { console.log("Emergency GPS error:", e); }
      }

    } catch (e) {
      console.log("Backend communication error:", e);
    } finally {
      isProcessingRef.current = false;
    }
  };

  // ── Audio playback ────────────────────────────────────────────────────────
  // IMPORTANT: timer is PAUSED for the entire duration of audio playback.
  // The 10s window is purely for the user to speak — not for the system
  // to play its response. Timer restarts only after audio fully finishes.
  const playAudioFromB64 = async (b64: string) => {
    const tmpPath = ((FileSystem as any).cacheDirectory as string) + 'vv_response.mp3';
    try {
      await FileSystem.writeAsStringAsync(tmpPath, b64, { encoding: 'base64' });

      // Pause the awake timer while audio plays
      clearAwakeTimer();

      await Audio.setAudioModeAsync({
        allowsRecordingIOS:      false,
        playsInSilentModeIOS:    true,
        staysActiveInBackground: true,
        shouldDuckAndroid:       false,
      });

      const { sound } = await Audio.Sound.createAsync(
        { uri: tmpPath },
        { shouldPlay: true, volume: 1.0 },
      );

      await new Promise<void>(resolve => {
        sound.setOnPlaybackStatusUpdate(async status => {
          if (status.isLoaded && status.didJustFinish) {
            await sound.unloadAsync();
            resolve();
          }
        });
      });

    } catch (e) {
      console.log("Playback error:", e);
    } finally {
      try {
        await Audio.setAudioModeAsync({
          allowsRecordingIOS:      true,
          playsInSilentModeIOS:    true,
          staysActiveInBackground: true,
          shouldDuckAndroid:       false,
        });
      } catch {}
      try { await FileSystem.deleteAsync(tmpPath, { idempotent: true }); } catch {}
      // Restart timer AFTER audio finishes — user now has a full fresh 10s to speak
      if (isAwakeRef.current) {
        resetAwakeTimer();
        console.log("Audio finished — fresh 10s timer started for user to speak.");
      }
    }
  };

  if (!permission?.granted) return <View style={styles.container} />;

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing="back" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black' },
  camera:    { flex: 1 },
});