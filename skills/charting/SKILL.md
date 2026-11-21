---
name: charting
description: Vẽ biểu đồ Chart.js trong báo cáo HTML — chọn loại biểu đồ, chuẩn bị dữ liệu bằng pandas, nhúng dữ liệu JSON từ CSV, fallback SVG khi không có mạng.
keywords: chartjs chart.js đồ thị visualization
---

# Charting

## When to Use

Khi báo cáo HTML cần **biểu đồ** để người đọc thấy xu hướng / so sánh / tỷ trọng trong một cái nhìn — không chỉ bảng số. Dùng kèm skill `agent-analytis` trong mission `agent_analytis_mission`: main agent vẽ biểu đồ cuối trong báo cáo HTML; **subagent chỉ chuẩn bị dữ liệu cho biểu đồ, không vẽ biểu đồ cuối**.

## Chart types — chọn đúng loại

| Loại | Dùng khi | Không dùng khi |
| --- | --- | --- |
| Line | Xu hướng theo thời gian (giá theo ngày/tháng, tăng trưởng) | So sánh hạng mục rời rạc |
| Bar | So sánh giữa các hạng mục (tỉnh, chuỗi, sản phẩm) | Hơn 12 cột — gộp nhóm lại trước |
| Pie / Donut | Tỷ trọng của một tổng (thị phần, cơ cấu) | Hơn 6 lát — dùng bar ngang thay thế |
| Scatter | Tương quan giữa 2 biến số | Chỉ 1 biến, hoặc dưới 10 điểm dữ liệu |

Heatmap không có sẵn trong Chart.js — cần ma trận màu thì vẽ bảng HTML/CSS hoặc SVG, không cố dùng Chart.js.

## Chart.js trong báo cáo HTML

- CDN: `<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>`, rồi mỗi biểu đồ một `<canvas id="...">` + `new Chart(ctx, {type, data, options})`.
- Dữ liệu nhúng dạng JSON trong block `<script>`, do **script Python sinh ra từ CSV** — không gõ tay số liệu vào HTML. Số trên biểu đồ phải khớp CSV (người đọc kiểm chứng được).
- Nhãn tiếng Việt; tiêu đề trục + đơn vị ghi rõ (vd. "Giá (USD/kg)"); mỗi biểu đồ có tiêu đề và dòng ghi nguồn số liệu.
- Responsive: `options: {responsive: true, maintainAspectRatio: false}` + khung `div` có chiều cao cố định.
- Tối đa 3 biểu đồ/báo cáo, trừ khi câu hỏi cần nhiều hơn. Không dùng ảnh PNG làm kênh chính.

## Chuẩn bị dữ liệu

- Aggregate bằng pandas trong sandbox trước khi vẽ: mỗi series tối đa ~50 điểm (nhiều hơn thì gộp theo tuần/tháng hoặc lấy top N).
- Xuất JSON cho biểu đồ kèm CSV dữ liệu gốc — JSON và CSV phải cùng một nguồn tính toán, không tính lại bằng tay.
- Thiếu dữ liệu thì vẽ đúng phần có dữ liệu và ghi rõ khoảng thiếu — không nội suy rồi trình bày như số thật.

## Fallback khi không có mạng

Sandbox không tải được CDN thì biểu đồ Chart.js sẽ trắng trơn. Trường hợp đó vẽ **SVG nội tuyến** (cột/đường tính bằng Python trong script sinh HTML) — báo cáo vẫn phải có biểu đồ khi mở offline. Kiểm tra bằng cách đọc lại HTML sau khi sinh: phải chứa `<canvas` + `new Chart(` (đường Chart.js) hoặc `<svg` (đường fallback).

## Subagent

Subagent chỉ trả dữ liệu đã aggregate (bảng + JSON) trong draft — không viết code Chart.js, không vẽ biểu đồ cuối. Main agent tổng hợp các draft rồi vẽ.
