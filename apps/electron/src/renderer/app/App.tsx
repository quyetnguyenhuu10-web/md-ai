import { useEffect } from "react";
import { AppShell } from "./layout/AppShell";
import { useConversationsStore } from "../features/conversations/store";
import { useChatStream } from "../features/chat/hooks/useChatStream";

export function App() {
  const initialize = useConversationsStore((state) => state.initialize);
  useChatStream();

  useEffect(() => {
    void initialize();
  }, [initialize]);

  return <AppShell />;
}
