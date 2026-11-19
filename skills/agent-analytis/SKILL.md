---
name: agent-analytis
description: Phân tích dữ liệu và nghiên cứu web bằng Tavily, chạy tính toán trong sandbox, xuất báo cáo HTML tự chứa (bảng + biểu đồ) kèm CSV và Markdown — dùng với mission `agent_analytis_mission`.
keywords: analytics research chart table html tavily phân tích nghiên cứu báo cáo biểu đồ bảng
---

# Agent Analytis

## When to Use

Khi yêu cầu cần **phân tích** (số liệu, file CSV, dữ liệu người dùng đưa)
hoặc **nghiên cứu web** (tin tức, giá cả, so sánh, tài liệu) và trả kết
quả dưới dạng **file, bảng, biểu đồ HTML** — không chỉ text.

## Steps

1. **Hiểu yêu cầu.** Dữ liệu gì, nguồn nào, output gì (file HTML? CSV?
   cả hai?). Thiếu thông tin thì hỏi lại trước khi làm — đừng đoán nguồn
   dữ liệu hay phạm vi.
2. **Thu thập — fan-out qua subagent.** Web: `tavily_search` (từ khóa
   hẹp, 5 kết quả) lấy danh sách URL. Rồi giao MỖI URL quan trọng cho
   1 subagent qua tool `task` (type `general-purpose`), gọi song song
   nhiều subagent trong cùng 1 message khi độc lập. Giao việc chặt:
   "Dùng `tavily_extract` đọc URL này, trích fact/số liệu/trích dẫn
   quan trọng + nguồn, trả lại bản draft tối đa ~800 token, và ghi
   draft vào `/output/drafts/<slug>.md` (hoặc `output/drafts/...`).
   Main agent KHÔNG tự
   `tavily_extract` toàn văn — chỉ đọc draft các subagent trả về rồi
   tổng hợp. Dữ liệu người dùng đưa: `read_file` / `ls` để xem file,
   không bịa nội dung file chưa đọc.
3. **Phân tích trong sandbox.** Làm việc từ draft của subagent + dữ
   liệu người dùng đưa (đọc lại draft trong `output/drafts/` khi cần,
   đừng giữ toàn bộ kết quả thô trong context). Viết script Python
   bằng `write_file` (pandas để tính toán, kiểm tra thiếu/sai dữ liệu
   trước khi kết luận), chạy bằng `execute`, đọc kết quả rồi mới kết
   luận. Tính sai thì sửa script và chạy lại — không sửa số liệu bằng
   tay cho khớp.
4. **Xuất kết quả — bắt buộc ra file.** Xem bảng dưới. Ghi MỌI file
   báo cáo vào thư mục `/output/` (ưu tiên đường dẫn tuyệt đối, ví dụ
   `/output/bao-cao.html`; `output/...` cũng được) — chỉ `.html` /
   `.csv` / `.md` (+ `.txt` / `.json` khi cần), mỗi file ≤ 256KB.
   Runtime tự tải các file trong
   `output/` ra ngoài và đưa vào kết quả; file nằm ngoài `output/`
   hoặc quá lớn sẽ KHÔNG tới được người đọc. Báo cáo HTML KHÔNG dán
   toàn bộ qua `write_file` — viết script Python nhỏ sinh HTML từ
   template + dữ liệu CSV rồi chạy trong sandbox; nội dung lớn chỉ
   sống trong sandbox, không vào context. Ghi rõ đường dẫn từng
   file trong câu trả lời cuối.
5. **Trả lời tóm tắt.** Kết luận chính + 2–3 con số quan trọng nhất +
   đường dẫn file + nguồn. Nguồn web nào cũng kèm URL.

## Output — định dạng bắt buộc

| File | Bắt buộc | Nội dung |
| --- | --- | --- |
| Báo cáo `.html` | Có | Bảng + biểu đồ, tự chứa, mở bằng trình duyệt không cần server |
| Dữ liệu `.csv` | Có (khi có số liệu) | Dữ liệu thô đã làm sạch để người đọc kiểm chứng lại |
| Tóm tắt `.md` | Có | Kết luận, số liệu chính, nguồn |

## Biểu đồ HTML

- Vẽ bằng **Chart.js qua CDN**; nếu sandbox không có mạng thì dùng
  **SVG nội tuyến**. Không dùng ảnh PNG làm kênh chính.
- Mỗi biểu đồ có: tiêu đề, nhãn trục + đơn vị, chú thích nguồn số liệu.
- Bảng HTML có `<thead>`/`<tbody>`, số liệu căn phải, đơn vị ghi rõ ở
  tiêu đề cột.

## Trích nguồn

- Khẳng định từ web kèm **URL** (trang đã đọc bằng `tavily_extract` —
  do subagent đọc rồi trả draft, không trích từ snippet search).
- Số liệu từ file kèm **tên file** (+ dòng/cột khi cần).
- Không có nguồn thì viết rõ "chưa có dữ liệu" — không bịa.

## What NOT to Do

- Không bịa số liệu, nguồn, URL hay kết quả tính toán.
- Không trích nguồn chưa đọc toàn văn.
- Không trả lời chỉ bằng text khi yêu cầu cần file / bảng / biểu đồ.
- Không chạy code ngoài sandbox, không gửi kết quả đi đâu khác ngoài
  file và câu trả lời của run này.
- Không tự `tavily_extract` toàn văn — giao cho subagent, main agent
  chỉ đọc draft.
- Không dán toàn bộ HTML qua `write_file` — sinh HTML bằng script
  trong sandbox.
