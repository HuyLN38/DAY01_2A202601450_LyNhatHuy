# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Khi temperature tăng từ 0.0 đến 1.8, phản hồi trở nên sáng tạo và đa dạng hơn nhưng cũng đi kèm với sự suy giảm tính hợp lý. Ở mức 0.0, câu trả lời rất an toàn và mang tính sự thật cao; ở mức 0.7, câu trả lời tự nhiên; nhưng khi lên đến 1.2 hoặc 1.8, phản hồi bắt đầu xuất hiện ảo giác (hallucination), sai ngữ pháp hoặc các từ ngữ ghép vào nhau một cách vô nghĩa và thiếu mạch lạc.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Trợ lý soạn thảo hợp đồng pháp lý cần temperature = 0.0 (hoặc rất thấp) để đảm bảo tính chính xác tuyệt đối, tránh sinh ra các điều khoản không hợp lệ hoặc sai sự thật. Ngược lại, trợ lý viết slogan quảng cáo cần sự đột phá và mới lạ nên có thể đặt temperature cao (ví dụ: 0.7 - 0.9) để kích thích sự sáng tạo.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Tổng lượng token output mỗi ngày là 20.000 * 2 * 500 = 20 triệu token. Với GPT-4o (giá 0.01$/1K token), chi phí là 200$/ngày; với GPT-4o-mini (giá 0.0006$/1K token), chi phí chỉ là 12$/ngày. Model lớn (GPT-4o) xứng đáng khi ứng dụng yêu cầu lập luận logic phức tạp hoặc phân tích dữ liệu chuyên sâu. Trong khi đó, model nhỏ (GPT-4o-mini) là lựa chọn tuyệt vời cho các tác vụ đơn giản như phân loại văn bản, tóm tắt nội dung hoặc chatbot chăm sóc khách hàng thông thường.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Phản hồi đầu tiên sử dụng các ngôn từ bay bổng, phép ẩn dụ mà không dùng thuật ngữ chuyên ngành. Phản hồi thứ hai tập trung vào định nghĩa kỹ thuật, thuật toán và có kèm theo đoạn code minh họa. Điều này cho thấy system prompt có thể điều khiển gần như toàn diện giọng văn, định dạng, mức độ chuyên sâu và cách thức trình bày của model.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Với 150 từ tiếng Việt, ước lượng thô cho ra khoảng 200 token, trong khi thực tế tiktoken có thể đếm ra 300-400 token (chênh lệch từ 50% đến 100%). Nếu dùng ước lượng thô, chúng ta sẽ dự toán thiếu ngân sách nghiêm trọng, nguyên nhân là do tiếng Việt sử dụng nhiều byte hơn trên mỗi ký tự/từ nên thường bị mã hóa thành nhiều token hơn so với tiếng Anh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Chatbot văn bản và trợ lý giọng nói (a, b) hưởng lợi nhiều nhất từ streaming vì người dùng không phải chờ toàn bộ văn bản dài sinh xong mới nhận được thông tin. Ngược lại, pipeline dịch tài liệu ban đêm (c) không cần streaming vì hệ thống chỉ cần lưu kết quả cuối cùng mà không cần tương tác ngay lập tức.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Exponential backoff giúp giảm tải đột ngột cho server đang bị sập hoặc quá tải bằng cách rải rác các nỗ lực kết nối lại theo thời gian thay vì dồn dập gửi request. Tuy nhiên, nếu hàng nghìn client cùng gặp lỗi ở một thời điểm, chúng vẫn có thể gửi request cùng lúc theo các bước nhảy số nhân. Kỹ thuật "jitter" thêm sự ngẫu nhiên vào thời gian chờ, giúp phân tán đều các yêu cầu và ngăn chặn triệt để tình trạng đồng bộ hóa ngoài ý muốn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt." 
> 1) Nếu xóa chữ "thân thiện", trợ lý có thể trả lời một cách khô khan, cứng nhắc và ít dùng các từ ngữ cảm thán hay emojis.
> 2) Nếu xóa chữ "ngắn gọn", model có thể tạo ra các câu trả lời dài dòng và giải thích lê thê thay vì đưa ra đáp án súc tích.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Nếu người dùng nêu một định nghĩa hoặc yêu cầu ở lượt đầu tiên (ví dụ: "Hãy đặt tên dự án là X"), sau đó thảo luận về các vấn đề khác trong 5 lượt tiếp theo, rồi hỏi "Dự án của tôi tên là gì?", trợ lý sẽ quên mất do lượt đầu tiên đã bị xóa khỏi history. Cách khắc phục là thay vì cắt cứng history, ta có thể kết hợp việc dùng model sinh ra bản tóm tắt các cuộc hội thoại cũ rồi nhúng vào phần system prompt.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)