from flask import Flask, request, jsonify
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

# 감정 라벨 매핑 (예시, 사용자의 실제 label_names 리스트로 수정)
label_names = [
    "돌돌대는", "좌절한", "짜증나는", "방어적인", "악의적인", "안달하는", "구역질 나는", "노여워하는", "성가신",  # 분노
    "실망한", "비통한", "후회되는", "우울한", "마비된", "염세적인", "눈물이 나는", "낙담한", "환멸을 느끼는",  # 슬픔
    "두려운", "스트레스 받는", "취약한", "혼란스러운", "당혹스러운", "회의적인", "걱정스러운", "조심스러운", "초조한",  # 불안
    "질투하는", "배신당한", "고립된", "충격 받은", "불우한", "희생된", "억울한", "괴로워하는", "버려진",  # 상처
    "고립된", "남의 시선 의식하는", "외로운", "열등감", "죄책감", "부끄러운", "혐오스러운", "한심한", "혼란스러운",  # 당황
    "감사하는", "사랑하는", "편안한", "만족스러운", "흥분되는", "느긋한", "안도하는", "신이 난", "자신하는"  # 기쁨
]

# 소분류 → 대분류 매핑
main_category_mapping = {
    0: "분노", 1: "분노", 2: "분노", 3: "분노", 4: "분노", 5: "분노", 6: "분노", 7: "분노", 8: "분노",
    9: "슬픔", 10: "슬픔", 11: "슬픔", 12: "슬픔", 13: "슬픔", 14: "슬픔", 15: "슬픔", 16: "슬픔", 17: "슬픔",
    18: "불안", 19: "불안", 20: "불안", 21: "불안", 22: "불안", 23: "불안", 24: "불안", 25: "불안", 26: "불안",
    27: "상처", 28: "상처", 29: "상처", 30: "상처", 31: "상처", 32: "상처", 33: "상처", 34: "상처", 35: "상처",
    36: "당황", 37: "당황", 38: "당황", 39: "당황", 40: "당황", 41: "당황", 42: "당황", 43: "당황", 44: "당황",
    45: "기쁨", 46: "기쁨", 47: "기쁨", 48: "기쁨", 49: "기쁨", 50: "기쁨", 51: "기쁨", 52: "기쁨", 53: "기쁨"
}

# 대분류 → 자연어 표현
main_natural = {
    "분노": "화났을 때",
    "슬픔": "슬플 때",
    "불안": "불안할 때",
    "상처": "상처받았을 때",
    "당황": "당황스러울 때",
    "기쁨": "기쁠 때"
}

# Flask 앱 시작
app = Flask(__name__)

# 모델 & 토크나이저 로드
model = AutoModelForSequenceClassification.from_pretrained("saved_emotion_model")
tokenizer = AutoTokenizer.from_pretrained("saved_emotion_model")
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False, device=0 if torch.cuda.is_available() else -1)

# 기본 루트
@app.route("/", methods=["GET"])
def home():
    return "Flask 서버 정상 작동 중! 감정 분석은 POST /predict 로 요청하세요."

# 테스트용 핑
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"})

# 감정 예측
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "텍스트가 없습니다."}), 400

        pred = pipe(text)[0]
        label = pred["label"]
        label_num = int(label.replace("LABEL_", ""))
        sub = label_names[label_num]
        main = main_category_mapping[label_num]
        main_natural_kor = main_natural[main]

        return jsonify({
            "emotion_sub": sub,
            "emotion_main": main,
            "emotion_main_natural": main_natural_kor
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 서버 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

