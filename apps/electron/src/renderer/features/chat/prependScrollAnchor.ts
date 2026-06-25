export interface PrependScrollSnapshot {
  scrollHeight: number;
  scrollTop: number;
}

export function capturePrependScrollSnapshot(element: Pick<HTMLDivElement, "scrollHeight" | "scrollTop">) {
  return {
    scrollHeight: element.scrollHeight,
    scrollTop: element.scrollTop,
  };
}

export function getRestoredScrollTopAfterPrepend(snapshot: PrependScrollSnapshot, nextScrollHeight: number) {
  const prependedHeight = Math.max(0, nextScrollHeight - snapshot.scrollHeight);
  return snapshot.scrollTop + prependedHeight;
}
