# MintDim Render Cache + Sliding Window Plan

## 0. Quyết định thiết kế hiện tại

Mô hình chính thức được chọn:

- Disk cache giữ cache bền cho các artifact render an toàn của message đã complete.
- Hot RAM cache chỉ giữ cache nóng quanh sliding render window / viewport.
- Renderer không giữ toàn bộ lịch sử hội thoại trong RAM.
- Renderer không giữ toàn bộ cache trong RAM.
- Native/storage giữ full conversation history.
- Markdown/LaTeX vẫn render bằng thư viện chuẩn.
- Không cache Markdown AST hoặc rendered Markdown output nếu chưa có đường an toàn từ thư viện.
- Nếu Markdown cache không chắc an toàn, chấp nhận để ReactMarkdown render lại mặc định.
- Cache chỉ tập trung vào thứ deterministic, khả thi, ít rủi ro:
  - Shiki code highlight output
  - measured heights
  - content hash
  - renderer/cache version
  - lightweight metadata

Nguyên tắc ngắn gọn:

Render window nhỏ để UI nhẹ.
Disk cache lớn để không làm lại việc cũ.
Hot cache nhỏ để không nổ RAM.
Markdown correctness quan trọng hơn cache Markdown liều.

---

## 1. Mục tiêu

Thiết kế pipeline chat renderer theo hướng:

- Markdown và LaTeX vẫn dùng thư viện chuẩn:
  - ReactMarkdown
  - remark-gfm
  - remark-math
  - rehype-katex
  - Shiki cho code block

- Không tự parse AST Markdown/LaTeX sâu để tránh render lệch thư viện.

- Message đã complete thì cache các artifact an toàn.

- Renderer chỉ giữ một cửa sổ message đủ lớn để scroll mượt.

- Native/storage giữ lịch sử hội thoại rất lớn.

- Khi user lướt lịch sử cũ hơn thì load dần theo chunk/cửa sổ cache.

- Cache bền nằm trên disk, không ép toàn bộ cache vào RAM.

---

## 2. Kiến trúc tổng thể

Native / DB:

- Giữ full history.
- Có thể giữ conversation rất lớn.
- Load message theo chunk bằng cursor/message id.
- Sau này có thể hỗ trợ loadBefore, loadAfter, loadAround.

Main / IPC:

- Cầu nối giữa renderer và native.
- Expose API:
  - loadRecentMessages
  - loadMessagesBefore
  - sau này: loadMessagesAfter
  - sau này: loadMessagesAround
  - sau này: getRenderCacheStats
  - sau này: clearRenderCache

Renderer Zustand:

- Chỉ giữ render window hiện tại.
- Không giữ toàn bộ conversation.
- Không giữ toàn bộ cache.
- Quản lý:
  - activeConversationId
  - window messages
  - oldestMessageId
  - newestMessageId
  - hasOlder
  - hasNewer
  - scrollTop
  - streamUpdateIntervalMs

VirtualMessageList:

- Chỉ mount vùng visible + overscan.
- Dùng measured height cache để tính layout ổn định.
- Khi thiếu height cache, fallback về estimatedHeight.

MessageItem:

- Message streaming cập nhật động.
- Message complete gần như bất biến.
- Message complete được cache artifact an toàn sau khi render/measure.

MarkdownRenderer:

- Giữ ReactMarkdown/remark/rehype/KaTeX.
- Không tự cache AST.
- Không tự cache rendered Markdown HTML toàn phần nếu chưa có đường an toàn.
- Chấp nhận render lại Markdown khi remount nếu cần.

CodeBlock:

- Dùng Shiki highlight.
- Cache highlighted HTML an toàn theo language + code + theme + rendererVersion.
- Cache này có thể nằm ở disk cache và hydrate vào hot RAM cache.

---

## 3. Hai tầng cache

### 3.1 Disk cache

Disk cache là cache bền.

Mục tiêu:

- Cache có thể lớn mà không làm nổ RAM.
- Cache tồn tại qua app restart nếu cần.
- Cache artifact của message complete.
- Không tự xoá âm thầm khi vượt soft limit.
- Chỉ cảnh báo và cho user clear thủ công sau này.

Disk cache có thể lưu:

- conversationId
- messageId
- contentHash
- rendererVersion
- code highlight cache entries
- measuredHeight
- estimatedHeight
- hasCodeBlock
- hasMath
- contentLength
- measuredAt
- updatedAt
- cacheSizeEstimate
- cache artifact version

Disk cache không lưu:

- DOM node sống
- React subtree sống
- Markdown AST tự chế
- rendered Markdown HTML toàn phần nếu chưa có threat model/sanitize rõ
- cache của message đang streaming như cache bền

### 3.2 Hot RAM cache

Hot cache là cache nhỏ trong RAM.

Mục tiêu:

- Scroll quanh viewport mượt.
- Không query disk liên tục cho vùng đang xem.
- Không tăng theo full history.
- Tự evict được theo sliding window.
- Nếu evict khỏi RAM thì disk cache vẫn còn.

Hot cache giữ:

- cache của message trong render window
- cache gần viewport/overscan
- cache của message đang streaming hoặc vừa complete
- recent code highlight HTML
- recent measured heights

Hot cache không giữ:

- toàn bộ conversation
- toàn bộ disk cache
- DOM node sống
- React subtree sống

---

## 4. Cache artifact được phép

### 4.1 Code highlight cache

Nên cache.

Input key:

- language
- code content hash
- Shiki theme
- Shiki version/cache version
- rendererVersion

Output:

- highlighted HTML đã sanitize theo pipeline hiện có
- language resolved
- code length
- generatedAt
- size estimate

Policy:

- Chỉ ghi disk cache bền khi message complete.
- Streaming code có thể cache tạm trong RAM.
- Nếu contentHash đổi, cache cũ không dùng.
- Nếu rendererVersion đổi, cache cũ không dùng.

### 4.2 Measured height cache

Nên cache.

Input key:

- conversationId
- messageId
- contentHash
- rendererVersion
- width bucket nếu cần

Output:

- measured height px
- measuredAt
- contentLength
- status when measured

Policy:

- Message complete đo xong thì ghi cache.
- Khi remount, dùng cached height trước estimatedHeight.
- Nếu width thay đổi nhiều, có thể invalidate theo width bucket.
- Nếu font/theme/layout version đổi, invalidate theo rendererVersion.

### 4.3 Lightweight metadata cache

Nên cache.

Có thể lưu:

- hasCodeBlock
- hasMath
- hasMarkdown
- isLongMessage
- contentLength
- contentHash
- tokenEstimate
- estimatedHeight
- renderedArtifactFlags

Policy:

- Nhẹ, an toàn.
- Có thể dùng để virtualizer và stats nhanh hơn.

### 4.4 Stream segment cache

Có thể cache nhưng không bắt buộc.

Chỉ cache nếu:

- splitProtectedStreamSegments ổn định.
- Cache theo contentHash.
- Không làm thay đổi render correctness.
- Không thay thế ReactMarkdown AST.

Nếu không chắc, bỏ qua.

---

## 5. Những thứ không cache vội

Không cache:

- Markdown AST.
- React element/subtree sống.
- DOM node đã render.
- Rendered Markdown HTML toàn phần.
- KaTeX HTML toàn message nếu chưa có đường an toàn từ thư viện.
- Output tự transform từ Markdown nếu có rủi ro lệch ReactMarkdown.

Lý do:

- Dễ lệch behavior của react-markdown/remark/rehype.
- Dễ sai khi plugin version đổi.
- Dễ dính sanitize/security.
- Dễ leak memory.
- Không đáng đổi correctness lấy performance.

Nguyên tắc:

Nếu Markdown caching không rõ an toàn, để ReactMarkdown render lại mặc định.

---

## 6. Sliding render window

Renderer chỉ giữ một cửa sổ message đang dùng.

Tham số đề xuất:

- LOAD_CHUNK_SIZE = 100
- WINDOW_TARGET_MESSAGES = 300
- WINDOW_MAX_MESSAGES = 600
- WINDOW_OVERSCAN_MESSAGES = 20

Có thể chỉnh sau theo thực tế.

### 6.1 Khi mở conversation

Flow:

1. User chọn conversation.
2. Renderer gọi loadRecentMessages(conversationId, WINDOW_TARGET_MESSAGES).
3. Store set render window bằng messages mới nhất.
4. Scroll về cuối.
5. Virtualizer chỉ mount viewport + overscan.
6. Với mỗi message complete:
   - check hot RAM cache
   - nếu miss, check disk cache
   - nếu disk hit, hydrate vào RAM
   - nếu disk miss, render mặc định rồi cache artifact an toàn sau khi đo/highlight

### 6.2 Khi scroll lên lịch sử cũ

Flow:

1. User scroll gần đầu window.
2. Lấy oldestMessageId.
3. Gọi loadMessagesBefore(conversationId, oldestMessageId, LOAD_CHUNK_SIZE).
4. Prepend older messages vào render window.
5. Preserve scroll anchor để màn hình không giật.
6. Hydrate hot cache từ disk nếu có.
7. Nếu message chưa có cache:
   - render mặc định
   - đo height
   - cache code highlight nếu có
   - ghi artifact an toàn xuống disk sau khi complete/stable
8. Nếu window vượt WINDOW_MAX_MESSAGES:
   - trim phía xa viewport
   - evict hot RAM cache tương ứng
   - không xoá disk cache

### 6.3 Khi scroll xuống vùng mới hơn

V0.1 có thể chưa cần loadAfter nếu app chủ yếu đi từ cuối lên cũ rồi xuống lại trong window.

Sau này nếu window bị trim cả hai phía:

- thêm loadMessagesAfter
- thêm loadMessagesAround
- dùng khi user nhảy search result hoặc timeline giữa history

### 6.4 Khi gửi message mới

Flow:

1. Append user message vào window.
2. Append assistant placeholder status streaming.
3. Assistant chunk tới đâu thì appendToMessage tới đó.
4. Streaming message render động.
5. Không ghi disk cache bền cho message streaming.
6. Khi assistant done:
   - replaceMessage thành status complete
   - tính contentHash
   - cache measured height
   - cache Shiki highlight output
   - ghi artifact an toàn xuống disk
7. Nếu user đang near bottom và window quá lớn:
   - trim phía đầu window
   - evict hot RAM cache phía xa
   - giữ disk cache

---

## 7. Scroll anchoring

Scroll anchoring là phần nguy hiểm nhất khi prepend/trim.

Trước khi thay đổi window:

- lấy message đầu tiên đang visible
- lưu messageId
- lưu offsetFromViewportTop

Sau khi prepend/trim:

- tìm lại DOM row theo messageId
- chỉnh scrollTop sao cho row đó vẫn ở offset cũ

Mục tiêu:

- load older không làm màn hình giật
- trim window không làm user bị teleport
- height cache giúp anchor ổn định hơn

---

## 8. Cache limit policy

### 8.1 RAM hot cache

RAM cache có giới hạn cứng.

Policy:

- Gắn với render window/viewport.
- Tự evict khi message rời xa window.
- Evict RAM không xoá disk.
- RAM cache không tăng theo toàn bộ history.

Gợi ý:

- giữ cache cho current window
- giữ thêm một ít overscan cache
- giữ cache của streaming/latest messages
- có thể giới hạn theo count hoặc estimated bytes

### 8.2 Disk cache

Disk cache có soft limit, không auto-delete.

Policy:

- Nếu dưới soft limit: im lặng.
- Nếu vượt soft warning: cảnh báo nhẹ.
- Nếu vượt strong warning: cảnh báo rõ.
- Nếu vượt danger warning: cảnh báo mạnh.
- Không tự xoá nếu user chưa bấm clear.

Ngưỡng gợi ý ban đầu:

- > 256 MB: warning nhẹ
- > 512 MB: warning rõ
- > 1 GB: danger warning

Sau này có thể chỉnh theo thực tế.

### 8.3 Manual cache controls

Sau này nên có trong Render Stats / Memory Inspector:

- disk cache size
- hot RAM cache estimate
- cached message count
- cached code block count
- measured height count
- cache hit rate
- cache miss rate
- warning state
- button Clear render cache
- button Open cache folder nếu cần

---

## 9. Renderer version / invalidation

Cache key phải có version.

rendererVersion nên bao gồm:

- Markdown renderer version
- KaTeX/remark/rehype behavior version
- CodeBlock/Shiki theme version
- CSS/layout version
- height measurement version

Nếu rendererVersion đổi:

- cache cũ không được dùng cho artifact nhạy layout/render.
- không nhất thiết xoá cache ngay.
- có thể chỉ coi là stale.
- disk warning có thể tính cả stale cache.

Ví dụ version concept:

- render-cache-v1
- code-highlight-v1
- height-layout-v1
- markdown-pipeline-v1

---

## 10. Phase 1: Low-risk render cache

Mục tiêu:

- Tối ưu thứ chắc chắn an toàn trước.
- Chưa đổi native storage lớn.
- Chưa cache Markdown AST/output.

Files dự kiến:

- src/renderer/features/chat/renderCache.ts
- src/renderer/features/chat/components/CodeBlock.tsx
- src/renderer/features/chat/components/VirtualMessageList.tsx
- src/renderer/features/chat/hooks/useMessageVirtualizer.ts
- có thể thêm stats trong RenderStatsPanel sau

Việc làm:

1. Thêm renderCache.ts.

Nội dung:

- hot RAM cache cho measured heights
- hot RAM cache cho code highlight HTML
- LRU hoặc bounded map nhẹ
- helper makeContentHash
- helper makeCodeHighlightKey
- helper get/set measured height
- helper get/set code highlight

2. CodeBlock dùng hot RAM cache.

Flow:

- resolve language
- make key từ language + code + theme + rendererVersion
- nếu hot cache hit: render ngay highlighted HTML
- nếu miss: gọi Shiki
- sau khi Shiki xong: set hot cache
- nếu message complete hoặc caller cho phép: queue ghi disk cache sau này

3. VirtualMessageList dùng measured height cache.

Flow:

- getHeight ưu tiên measuredHeightCache
- nếu không có, dùng measuredHeights local
- nếu không có, dùng message.layout.estimatedHeight
- khi row đo được height mới, set cache

4. Complete row giảm ResizeObserver.

Policy:

- streaming message observe liên tục
- complete message observe đến khi stable rồi disconnect
- nếu height đổi rõ thì cập nhật cache

Verify:

- npx tsc --noEmit
- npm run test:streaming
- npm run test:contracts

Manual test:

- stream message có Markdown/LaTeX/code
- scroll đi rồi quay lại code block
- kiểm tra không bị vỡ render
- kiểm tra height không nhảy mạnh

---

## 11. Phase 2: Disk render cache

Mục tiêu:

- Cache bền không làm nổ RAM.
- Hot cache hydrate từ disk.
- Không auto-delete disk cache.

Thiết kế ban đầu:

- Disk cache có thể nằm trong app data dir.
- Có thể là SQLite table hoặc files trong conversation cache folder.
- Ưu tiên SQLite/native nếu đã có storage_sqlite.
- Renderer không trực tiếp ghi file tùy tiện nếu đã có native sidecar phù hợp.

API có thể cần:

- renderCache.getArtifacts(conversationId, messageIds)
- renderCache.putArtifact(artifact)
- renderCache.getStats()
- renderCache.clear()
- renderCache.getWarnings()

Artifact ban đầu:

- measured height
- code highlight HTML
- content hash
- renderer version
- metadata

Không artifact:

- Markdown AST
- full rendered Markdown HTML
- DOM/React node

Flow hydrate:

1. Virtual window load messages.
2. Renderer hỏi disk cache artifact cho message ids trong window.
3. Artifact trả về được đưa vào hot RAM cache.
4. Components đọc hot cache.
5. Miss thì render bình thường.
6. Sau render/measure, queue putArtifact xuống disk.

Verify:

- npx tsc --noEmit
- npm run test:streaming
- npm run test:contracts
- build native nếu đụng native

---

## 12. Phase 3: Bounded renderer window

Mục tiêu:

- visibleMessages không phình mãi.
- Store chỉ giữ render window.
- DOM/store/RAM ổn định với conversation dài.

Thay model store dần:

- activeConversationId
- messages
- oldestMessageId
- newestMessageId
- hasOlder
- hasNewer
- isLoadingOlder
- isLoadingNewer
- scrollTop

Policy:

- replaceWindow
- prependOlderChunk
- appendNewMessages
- trimWindowAroundViewport
- clearConversationWindow
- hydrateCacheForWindow

Cẩn thận:

- preserve scroll anchor khi prepend
- preserve scroll anchor khi trim
- không trim message đang streaming
- không trim vùng gần viewport
- không trim nếu đang interaction nhạy

Verify:

- npx tsc --noEmit
- npm run test:streaming
- npm run test:contracts

Manual test:

- conversation dài
- scroll lên load older
- scroll xuống lại
- gửi message mới
- streaming không giật
- cache hydrate không làm flicker

---

## 13. Phase 4: Paging API mở rộng

Hiện tại v0.1 có thể dùng:

- loadRecentMessages
- loadMessagesBefore

Sau này thêm:

- loadMessagesAfter
- loadMessagesAround

Cần update:

- src/contracts/ipc-contracts/conversation.contract.ts
- src/preload/index.ts
- src/main/modules/conversations/conversation.ipc.ts
- src/main/modules/conversations/ConversationService.ts
- native sidecar protocol
- native message store

Use cases:

- user nhảy search result
- user ở giữa history
- renderer window bị trim cả hai phía
- timeline/minimap
- restore session quanh anchor cũ

---

## 14. Phase 5: Storage cho history rất lớn

Mục tiêu:

- Conversation có thể rất lớn.
- Renderer vẫn nhẹ.
- Native load chunk nhanh.
- Search/context không phụ thuộc renderer holding history.

Storage nên có:

- index theo conversationId
- index theo ordinal/createdAt
- index message id
- paging before/after/around
- metadata table
- cache artifact table/folder
- WAL/checkpoint policy

Không gửi blob lớn qua IPC nếu không cần.

---

## 15. Render stats / warnings

RenderStatsPanel hoặc MemoryInspector nên hiển thị:

- visible message count
- render window count
- hot cache estimated size
- disk cache estimated size
- cached complete message count
- code highlight cache count
- measured height cache count
- disk cache warning level
- stream FPS/update interval
- cache hit/miss nếu có

Warning copy gợi ý:

- Render cache is growing but remains on disk.
- Hot memory cache is bounded to the current render window.
- Disk cache exceeded soft limit. No cache was deleted automatically.
- You can clear render cache manually if needed.

---

## 16. Definition of Done cho v0.1

Đạt khi:

- Markdown/LaTeX/code vẫn render đúng.
- Không custom sâu Markdown AST.
- Code block không bị Shiki highlight lại liên tục khi scroll qua lại.
- Message complete không bị ResizeObserver đo vô hạn.
- Height cache giúp scroll ổn định hơn.
- Renderer không giữ message array phình mãi.
- Hot RAM cache bị giới hạn theo window.
- Disk cache giữ artifact bền.
- Disk cache vượt soft limit thì cảnh báo, không tự xoá.
- Load older chạy theo chunk.
- UI vẫn mượt với hội thoại dài.
- Không cần renderer giữ toàn bộ lịch sử hội thoại.
- Không rerender vô ích toàn bộ tin nhắn khi đang stream nhưng cũng ko làm mất tính chuẩn của thư viện. chấp nhận rerender nhưng cái nào đã ổn thì dừng rerender, tránh rerender vô ích.
---

## 17. Nguyên tắc patch

- Patch nhỏ theo phase.
- Không đập một lần quá lớn.
- Ưu tiên correctness trước performance.
- Không cache Markdown nếu không chắc an toàn.
- Không giữ DOM/React node sống làm cache.
- Mọi PowerShell patch phải crash-safe.
- Có backup/rollback.
- Không dùng exit trong PowerShell patch.
- Verify bằng:
  - npx tsc --noEmit
  - npm run test:streaming
  - npm run test:contracts
- Nếu đụng native thì build/test native riêng.

---

## 18. Tóm tắt một câu

Disk cache là kho nhớ dài hạn cho artifact render an toàn.
Hot RAM cache là bàn làm việc nhỏ quanh viewport.
Sliding render window là cửa sổ nhìn vào lịch sử lớn.
Markdown/LaTeX giữ đúng bằng thư viện chuẩn, không cache liều.