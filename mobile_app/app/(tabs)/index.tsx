import { CameraView, useCameraPermissions } from "expo-camera";
import { useEffect } from "react";
import { View, StyleSheet } from "react-native";

export default function HomeScreen() {

  const [permission, requestPermission] = useCameraPermissions();

  useEffect(() => {
  if (!permission) {
    requestPermission();
  }
}, [permission, requestPermission]);

  if (!permission?.granted) {
    return <View style={{ flex: 1, backgroundColor: "black" }} />;
  }

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing="back" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
});