# Render Cache + Sliding Window Checklist

Nguon: `apps/electron/plan_render_cacache-v2.md`
Ngay khao sat: 2026-06-24

Muc tieu file nay: bien plan thanh checklist co the patch theo tung buoc nho, dua tren code hien co trong `apps/electron`.

## 0. Snapshot code hien tai

- [x] Hot RAM cache da co trong `src/renderer/features/chat/renderCache.ts`: cache Shiki HTML, cache measured height, bounded theo count/estimated bytes.
- [x] Key cache da co version co ban: `render-cache-v1`, `code-highlight-v1`, `message-height-v1`.
- [x] `CodeBlock.tsx` da doc hot cache truoc, doc disk cache khi non-streaming, fallback Shiki, sanitize HTML, va chi persist khi non-streaming; streaming code khong goi Shiki moi token.
- [x] `VirtualMessageList.tsx` da dung height cache, hydrate height tu disk, va disconnect `ResizeObserver` sau khi message complete settle.
- [x] Disk render cache da co adapter main-side `src/main/modules/utilities/RenderCacheDiskService.ts`, luu artifact JSON trong app data dir.
- [x] IPC/preload da co `getRenderCacheArtifacts`, `putRenderCacheArtifact`, `getRenderCacheDiskStats`, `clearRenderCache`.
- [x] `RenderStatsPanel.tsx` da hien hot cache size, Shiki entries, height entries, disk size, artifact count, warning level.
- [x] Native sidecar da co `message.loadRecent` va `message.loadBefore`.
- [x] Native SQLite da co `message_layout_cache` cho estimated layout metadata va index `messages(conversation_id, created_at)`.
- [x] Markdown/LaTeX van render qua `ReactMarkdown`, `remark-gfm`, `remark-math`, `rehype-katex`; chua cache Markdown AST/full rendered HTML.
- [x] Markdown streaming text da split theo blank-line block va co streaming render window de prefix on dinh khong parse/render lai moi token.
- [x] Chat store da co render window state: `oldestMessageId`, `newestMessageId`, `hasOlder`, `hasNewer`, `isLoadingNewer`.
- [x] Streaming delta dung overlay `streamingMessageContentById`, khong mutate `visibleMessages` moi token.
- [x] Load chunk khong con hard-code `50`; da co bo tham so plan: `LOAD_CHUNK_SIZE = 100`, `WINDOW_TARGET_MESSAGES = 300`, `WINDOW_MAX_MESSAGES = 600`, `WINDOW_OVERSCAN_MESSAGES = 20`.
- [x] `useScrollAnchor.ts` da xu ly stick-to-bottom va `VirtualMessageList.tsx` da restore scrollTop sau prepend older chunk.
- [ ] Chua co trim window quanh viewport, chua evict hot cache theo message bi trim.
- [ ] Chua co `loadMessagesAfter` / `loadMessagesAround`.
- [x] API/UI `clearRenderCache` da co trong contract/preload/main service/IPC va `RenderStatsPanel`.
- [ ] Chua co test rieng day du cho virtualizer, scroll anchor, paging, window trim; render cache, disk cache, va chat store da co test co ban.

## 1. Boundary va architecture guardrails

- [ ] Renderer chi goi cache disk qua `window.chatAPI`; khong import `src/main`, `electron`, `node:*`, `fs`.
- [ ] Main/utilities la noi giu adapter disk cache; renderer khong tu ghi file.
- [ ] Neu chuyen disk cache sang native SQLite thi di qua contracts/preload/main service, khong de renderer goi native truc tiep.
- [ ] Contract types nam trong `src/contracts/**`; khong import renderer/main implementation vao contracts.
- [ ] Khong tao module ten mo ho nhu `utils`, `helpers`, `shared`, `manager`.
- [ ] Khong cache Markdown AST, React subtree, DOM node, full rendered Markdown HTML neu chua co threat model va sanitize contract ro rang.
- [ ] Moi API paging/cache moi phai cap nhat `tests/contracts/electron_contract.yaml` neu boundary thay doi.

## 2. Phase 1 - Hoan thien low-risk hot render cache

### 2.1 `renderCache.ts`

- [x] Tach cac constant window/cache thanh ten ro nghia trong dung root renderer: `renderWindowPolicy.ts`.
- [ ] Them layout/theme/width bucket vao message height cache key neu height phu thuoc width, font, theme, CSS layout.
- [ ] Ghi ro version component: Markdown pipeline version, Shiki theme version, CSS/layout version, height measurement version.
- [x] Khi evict code highlight entry, xoa luon metadata trong `codeHighlightKeyMetadata` de tranh map metadata tang mai.
- [x] Dat bound hoac cleanup cho `persistedArtifactSignatures`.
- [ ] Them helper evict hot cache theo danh sach message id khi window trim.
- [ ] Chong ghi disk lap lai sau khi hydrate disk height neu payload/signature khong doi.
- [x] Validate payload khi doc disk artifact: `html` phai string, `height` phai finite positive number, size khong vuot limit.
- [ ] Them cache hit/miss counters trong hot stats neu can hien o Render Stats.

### 2.2 `CodeBlock.tsx`

- [x] Dam bao streaming code block khong persist disk cache ben vung.
- [ ] Dam bao completed code block cache theo `language + code content hash + theme + rendererVersion`.
- [ ] Kiem tra fallback unknown language ve `text` khong tao cache key sai language.
- [ ] Kiem tra `sanitizeShikiHtml` loai event handler, `javascript:` URL, tag nguy hiem.
- [ ] Xem xet batch/preload disk artifacts cho code block trong current window de tranh IPC waterfall neu message co nhieu code block.
- [ ] Test scroll qua lai code block complete khong goi Shiki lai khi hot/disk cache hit.

### 2.3 `VirtualMessageList.tsx` va `useMessageVirtualizer.ts`

- [x] Doi overscan hien tai `6` thanh policy ro rang, gan voi `WINDOW_OVERSCAN_MESSAGES`.
- [ ] Height lookup order can giu: measured local -> hot cache -> disk hydrated -> `message.layout.estimatedHeight` -> fallback.
- [ ] Complete row chi observe den khi stable roi disconnect; streaming row observe lien tuc.
- [ ] Khi message complete va height stable, persist height cache mot lan hop ly.
- [ ] Khi conversation doi, clear measured local map va hydrate lai theo window moi.
- [ ] Test virtualizer tinh dung `totalHeight`, range visible, overscan, fallback height.

## 3. Phase 2 - Disk render cache

- [ ] Quyet dinh chot V0.1 giu JSON file store hay chuyen sang native SQLite.
- [x] Neu giu JSON file store, khoa adapter o main `RenderCacheDiskService` va ghi ro khong phai renderer-owned file I/O.
- [x] Them `clearRenderCache` vao `UtilitiesApiContract`, preload, `UtilitiesService`, `RenderCacheDiskService`, IPC handler.
- [x] Them UI button Clear render cache trong `RenderStatsPanel` hoac `MemoryInspector`.
- [ ] Them `getRenderCacheWarnings` neu warning copy can phong phu hon `warningLevel`.
- [ ] Khong auto-delete disk cache khi vuot soft/strong/danger limit.
- [x] Disk cache clear phai la thao tac user-triggered, crash-safe, khong xoa folder ngoai render cache root.
- [ ] `RenderCacheDiskService` can reject artifact key/payload qua lon, payload sai shape, kind khong support.
- [x] Can test put/read va read invalid JSON khong crash.
- [ ] Can test warning thresholds: ok, soft, strong, danger.
- [ ] Can test `getArtifacts` dedupe keys, limit `MAX_KEYS_PER_READ`, ignore key qua dai.

## 4. Phase 3 - Bounded renderer window

### 4.1 Store shape

- [x] Doi `src/renderer/features/chat/store.ts` tu `visibleMessages` don gian thanh render window state ro rang.
- [x] Them `oldestMessageId`.
- [x] Them `newestMessageId`.
- [x] Them `hasOlder`.
- [x] Them `hasNewer`.
- [x] Them `isLoadingNewer`.
- [ ] Giu `scrollTop`.
- [ ] Giu `streamUpdateIntervalMs`.
- [x] Them boundary options cho action `replaceVisibleMessages(messages, boundaryState)`.
- [x] Them boundary options cho action `prependMessages(messages, { hasOlder })`.
- [x] Them boundary options cho action `appendMessages(messages, { hasNewer })`.
- [ ] Them action `trimWindowAroundViewport(anchor/visibleRange)`.
- [x] Reset conversation window khi `setActiveConversation` / `clearActiveConversation`.
- [ ] Them action `hydrateCacheForWindow(messages)` neu can tach khoi component.

### 4.2 Conversation selection

- [x] `src/renderer/features/conversations/store.ts` goi `loadRecentMessages(conversationId, WINDOW_TARGET_MESSAGES)` thay vi `50`.
- [x] Khi select conversation, set window messages va cursor state cung luc.
- [ ] Scroll ve cuoi sau khi window moi render.
- [x] Neu recent load tra it hon target, set `hasOlder = false`.
- [x] Neu co du target, tam set `hasOlder = true` cho den khi load before tra empty.

### 4.3 Load older

- [x] `VirtualMessageList.tsx` dung `LOAD_CHUNK_SIZE` thay vi `50`.
- [x] Khi scroll gan dau va `hasOlder`, goi `loadMessagesBefore(activeConversationId, oldestMessageId, LOAD_CHUNK_SIZE)`.
- [x] Neu older chunk empty, set `hasOlder = false` de khong spam IPC.
- [x] Prepend phai dedupe id.
- [x] Prepend phai preserve scroll anchor theo previous scrollTop + prepended height delta.
- [ ] Sau prepend, hydrate cache cho chunk moi.
- [ ] Neu window vuot `WINDOW_MAX_MESSAGES`, trim phia xa viewport.

### 4.4 Append/send/streaming

- [ ] Khi user send, append user message va assistant placeholder vao window.
- [ ] Khong trim message dang streaming.
- [ ] Khi assistant done, replace message complete, cache height/code artifact sau khi render stable.
- [ ] Neu user dang near bottom va window qua lon, trim phia dau window.
- [ ] Neu user khong near bottom, append khong teleport scroll.
- [x] Streaming delta batch bang `requestAnimationFrame` hien da co; native persistence da batch, khong ghi SQLite moi token.
- [x] Streaming delta chi re-render message row dang stream; visible window, virtualizer, hydrate cache, va side panels khong bi keo theo moi token.
- [x] Markdown realtime khi streaming dung sliding window: stable prefix commit rieng, live tail render theo active block context.
- [x] Live Markdown phoi hop voi `ReactMarkdown`: block nho render bang thu vien chuan, block dai dung active context + inline window va van render chunk/suffix bang `ReactMarkdown`.
- [x] Streaming overlay buffer duoc chunk den som truoc khi assistant placeholder vao visible window.
- [x] Render Stats FPS sample duoc throttle, khong update side panel moi animation frame.

### 4.5 Trim va hot cache eviction

- [ ] Trim khong duoc cat vung visible + overscan.
- [ ] Trim khong duoc cat message dang streaming.
- [ ] Trim khong duoc cat message user dang focus/select/copy neu co interaction nhay.
- [ ] Evict measured local heights cua message roi khoi window.
- [ ] Evict hot RAM cache cua message roi xa window neu co helper theo message id.
- [ ] Khong xoa disk cache khi trim.
- [ ] Render Stats hien `render window count` rieng voi `visible/mounted count` neu co.

## 5. Phase 4 - Paging API mo rong

- [ ] Them `loadMessagesAfter` vao `src/contracts/ipc-contracts/conversation.contract.ts`.
- [ ] Them `loadMessagesAround` vao contract neu can jump search/timeline.
- [ ] Cap nhat `src/preload/index.ts`.
- [ ] Cap nhat `src/main/modules/conversations/conversation.ipc.ts`.
- [ ] Cap nhat `src/main/modules/conversations/ConversationService.ts`.
- [ ] Cap nhat native JSON-RPC method `message.loadAfter`.
- [ ] Cap nhat native JSON-RPC method `message.loadAround`.
- [ ] Cap nhat `MessageStore`.
- [ ] Cap nhat `StorageSQLite`.
- [ ] Them stable cursor: uu tien `ordinal`; toi thieu dung cap `(created_at, id)`.
- [ ] Sua `loadBefore` hien tai, vi `created_at < before.created_at` co the bo sot message trung timestamp.
- [ ] Query can order on stable cursor, khong chi `created_at`.
- [ ] Them index phu hop cho cursor moi.
- [ ] Search result jump dung `loadMessagesAround(messageId, beforeLimit, afterLimit)`.

## 6. Phase 5 - Storage/history lon

- [ ] Danh gia per-conversation DB hien tai co du cho history rat lon hay can table/index bo sung.
- [ ] Them ordinal/message sequence trong schema neu chon cursor theo ordinal.
- [ ] Them migration an toan cho conversation DB cu.
- [ ] Xem lai `PRAGMA journal_mode=DELETE`; neu can WAL thi co policy ro cho checkpoint/backup.
- [ ] Khong gui blob lon qua IPC neu UI chi can metadata/window chunk.
- [ ] Search/context khong phu thuoc renderer giu full history.
- [ ] Can native tests cho paging before/after/around voi conversation dai.

## 7. Render stats va controls

- [ ] Hien `render window count`.
- [ ] Hien mounted/virtualized row count neu lay duoc tu virtualizer.
- [ ] Hien hot cache estimated bytes.
- [ ] Hien disk cache estimated bytes.
- [ ] Hien code highlight cache count.
- [ ] Hien measured height cache count.
- [ ] Hien cache hit/miss neu them counters.
- [ ] Hien disk warning copy dung policy: warning nhung khong auto-delete.
- [x] Them Clear render cache button.
- [x] Sau clear, refresh disk stats va hot stats.

## 8. Test checklist

- [x] `npx tsc --noEmit`.
- [x] `npm run test:streaming`.
- [x] `npm run test:stream-rendering`.
- [x] `npm run test:stream-live-markdown`.
- [x] `npm run test:stream-window`.
- [x] `npm run test:chat-store`.
- [x] `npm run test:chat-streaming`.
- [x] `npm run test:contracts`.
- [x] `npm run test:render-cache-disk`.
- [x] Test unit cho `renderCache.ts`: key version/content invalidation, bound/eviction, payload validation.
- [ ] Test unit cho `useMessageVirtualizer.ts`: range, total height, overscan, cached/fallback height.
- [ ] Test renderer behavior cho `CodeBlock`: hot hit, disk hit, streaming no persist, unknown language fallback.
- [ ] Test main service `RenderCacheDiskService`: put/read, invalid JSON, clear da co; warning thresholds va atomic-failure simulation chua khoa rieng.
- [x] Test store window reducers: replace, prepend, append, dedupe, hasOlder/hasNewer basics.
- [x] Test streaming overlay store: append delta khong replace visible window, done commit va clear overlay.
- [x] Test streaming overlay store: chunk som duoc buffer truoc khi placeholder visible.
- [x] Test streaming markdown window: stable prefix reuse, live tail update, fenced code incomplete/complete.
- [x] Test streaming live markdown context: active block classify, inline chunk commit/reuse, avoid commit unclosed strong marker.
- [ ] Test store window reducer trim.
- [x] Test unit cho prepend scroll anchor math.
- [ ] Test scroll anchor logic bang DOM/jsdom hoac Playwright neu co harness.
- [ ] Test native paging voi duplicate `created_at` de khoa bug cursor.
- [ ] Neu dung native C++ thay doi: build native va chay smoke sidecar.

## 9. Manual QA checklist

- [ ] Mo conversation dai, load recent window, scroll mượt.
- [ ] Scroll len dau window, load older khong giat viewport.
- [ ] Scroll xuong lai sau prepend, anchor van dung vi tri cu.
- [ ] Gui message moi khi dang near bottom, UI stick bottom hop ly.
- [ ] Gui message moi khi dang doc lich su cu, UI khong teleport.
- [ ] Markdown render dung.
- [ ] LaTeX inline/display render dung.
- [ ] Code block complete khong highlight lai lien tuc khi scroll qua lai.
- [ ] Streaming code block hien raw suffix hop ly, khong persist disk truoc khi complete.
- [ ] Restart app, disk cache hydrate height/code artifact.
- [ ] Disk cache vuot threshold chi warning, khong tu xoa.
- [ ] Clear render cache xoa dung render cache folder va UI stats refresh.

## 10. Suggested patch order

- [x] Patch 1: fix hot cache housekeeping, payload validation, cache key version notes, tests nho cho `renderCache.ts`.
- [x] Patch 2: add `clearRenderCache` API/UI va tests cho `RenderCacheDiskService`.
- [x] Patch FPS: batch native message persistence khi streaming, streaming code block khong Shiki moi token, split markdown streaming theo block on dinh.
- [x] Patch FPS 2: streaming overlay theo message id + sliding Markdown render window, tranh render lan moi token.
- [x] Patch FPS 3: hybrid live Markdown renderer dung prefix/context; `ReactMarkdown` chi render micro-window/chunk nho khi stream, full pipeline render lai khi message complete.
- [x] Patch 3: introduce render window constants va store state cursors without trim.
- [x] Patch 4: load older voi `hasOlder`, chunk size, dedupe, disk height hydrate.
- [x] Patch 5: implement prepend scroll anchoring theo scrollHeight delta.
- [ ] Patch 6: implement trim around viewport va hot cache eviction theo message id.
- [ ] Patch 7: add stable cursor/native paging fix cho duplicate timestamp.
- [ ] Patch 8: add `loadAfter/loadAround` end-to-end.
- [ ] Patch 9: broaden stats/manual controls va long-conversation QA.

## 11. Definition of Done v0.1

- [ ] Renderer khong giu full conversation history.
- [ ] Renderer window bi bound theo policy ro rang.
- [ ] Hot RAM cache bi bound va co eviction.
- [ ] Disk cache giu artifact an toan: code highlight HTML, measured height, lightweight metadata.
- [ ] Disk cache warning khong auto-delete.
- [ ] Markdown/LaTeX correctness giu theo thu vien chuan.
- [ ] Complete message khong bi `ResizeObserver` do vo han.
- [ ] Scroll prepend/trim khong teleport viewport.
- [ ] Paging before khong bo sot message trung timestamp.
- [ ] Tests contract/streaming/typecheck pass.
