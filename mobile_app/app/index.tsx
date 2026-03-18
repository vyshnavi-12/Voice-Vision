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

  // Helper functions to manage timer and sleep/awake states

  const clearAwakeTimer = () => {
    if (awakeTimerRef.current) {
      clearTimeout(awakeTimerRef.current);
      awakeTimerRef.current = null;
    }
  };

  // Restart the 10-second wake timer. If system is still processing, reschedule it
  const resetAwakeTimer = () => {
    clearAwakeTimer();
    awakeTimerRef.current = setTimeout(async () => {
      if (isProcessingRef.current) {
        resetAwakeTimer();
        return;
      }
      await goToSleep("10s inactivity timeout");
    }, AWAKE_TIMEOUT_MS);
  };

  // Convert text to speech and play it
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

  // Stop being awake - turn off timer and say goodbye
  const goToSleep = async (reason: string) => {
    if (!isAwakeRef.current) return;
    console.log(`Going to sleep — reason: ${reason}`);
    isAwakeRef.current = false;
    clearAwakeTimer();
    await speakText("Going to sleep.");
    isSystemBusyRef.current = false;
    console.log("Obstacle detection resumed.");
  };

  // Check every few seconds for obstacles in front using camera
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


  // Keep checking if camera is aligned for text reading. Automatically retry every 2.5s for up to 30s
  const runOcrGuidanceLoop = async () => {
    const MAX_ATTEMPTS  = 12;  // 12 x 2.5s = 30s max
    const RETRY_DELAY_MS = 2500;

    for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
      console.log(`OCR guidance loop — attempt ${attempt + 1}/${MAX_ATTEMPTS}`);

      // Wait a bit then take a new photo
      await new Promise(r => setTimeout(r, RETRY_DELAY_MS));

      // Capture current camera frame
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

        // Play the audio message
        if (data.audio) await playAudioFromB64(data.audio);

        if (!data.ocr_needs_guidance) {
          // Camera is good and text was read - done
          await goToSleep("OCR completed");
          return;
        }
        // Camera still not aligned - keep trying

      } catch (e) {
        console.log("OCR guidance loop error:", e);
      }
    }

    // Took too long - tell user and stop trying
    await speakText("Could not read the text. Please try again.");
    await goToSleep("OCR guidance timeout");
  };

  // Continuously record audio in 3.5 second chunks and send to backend for processing
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

  // Send recorded audio and camera frame to backend for analysis
  const sendToBackend = async (uri: string, isAwake: boolean) => {
    if (isAwake) isProcessingRef.current = true;
    try {
      let photoBase64: string | null = null;
      if (cameraRef.current) {
        try {
          const photo = await cameraRef.current.takePictureAsync({ 
            base64: true, 
            quality: isAwake ? 0.8 : 0.3   // high quality when processing commands
          });
          photoBase64 = photo?.base64 ?? null;
        } catch {}
      }

      const audioBase64 = await FileSystem.readAsStringAsync(uri, { encoding: 'base64' });

      const res = await fetch(`${BACKEND_URL}/process_command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audio: audioBase64, image: photoBase64, is_awake: isAwake }),
      });

      if (res.status === 204) return;
      if (!res.ok) { console.log("Backend error:", res.status); return; }

      const data = await res.json();
      console.log("Intent:", data.intent, "| needs_more_info:", data.needs_more_info, "| ocr_guidance:", data.ocr_needs_guidance);

      // User said wake word - switch from sleeping to awake mode
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
        // User said stop command
        await goToSleep("user said STOP");

      } else if (data.ocr_needs_guidance) {
        // Camera not aligned for text reading - keep auto-checking until aligned
        console.log("OCR guidance — starting auto frame loop...");
        await runOcrGuidanceLoop();

      } else if (data.needs_more_info) {
        // Backend is asking a follow-up question
        console.log("Follow-up question — user has fresh 10s after audio ends.");

      } else if (data.intent && data.intent !== 'UNKNOWN' && data.intent !== 'IDLE') {
        // Command was processed successfully - go to sleep
        await goToSleep("command completed");

      }

      // AFTER — only fires after user confirms who to send to (needs_more_info is false)
      if (data.intent === 'EMERGENCY_REQUESTED' && !data.needs_more_info) {
        try {
          const loc = await Location.getCurrentPositionAsync({});
          const emergRes = await fetch(`${BACKEND_URL}/trigger_emergency`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              location: { latitude: loc.coords.latitude, longitude: loc.coords.longitude },
            }),
          });
          if (emergRes.ok) {
            const emergData = await emergRes.json();
            if (emergData.audio) await playAudioFromB64(emergData.audio);
          }
        } catch (e) { console.log("Emergency GPS error:", e); }
      }
    } catch (e) {
      console.log("Backend communication error:", e);
    } finally {
      isProcessingRef.current = false;
    }
  };

  // Play audio response. Timer pauses while audio plays and restarts after it finishes
  const playAudioFromB64 = async (b64: string) => {
    const tmpPath = ((FileSystem as any).cacheDirectory as string) + 'vv_response.mp3';
    try {
      await FileSystem.writeAsStringAsync(tmpPath, b64, { encoding: 'base64' });

      // Stop timer while audio is playing
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
          allowsRecordingIOS: true,
          playsInSilentModeIOS: true,
          staysActiveInBackground: true,
          shouldDuckAndroid: false,
        });
        await new Promise(r => setTimeout(r, 300)); // wait for iOS session to activate
      } catch {}
      try { await FileSystem.deleteAsync(tmpPath, { idempotent: true }); } catch {}
      // Restart timer after audio finishes so user has a fresh 10 seconds to speak
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