import { Stack } from 'expo-router';

export default function RootLayout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      {/* This automatically finds your index.tsx in the app folder */}
      <Stack.Screen name="index" />
    </Stack>
  );
}