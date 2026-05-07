import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from openai import OpenAI


app = FastAPI(
    title="ExamPrep Agent API",
    description="A beginner AI agent that creates study plans, quizzes, and feedback.",
    version="1.0.0"
)


class Topic(BaseModel):
    name: str
    confidence: int = Field(ge=1, le=5)


class StudentProfile(BaseModel):
    student_name: str
    subject: str
    exam_date: str
    topics: List[Topic]
    study_hours_per_day: float
    grade_goal: str
    preferred_study_style: str
    biggest_worry: str


class QuizRequest(BaseModel):
    subject: str
    topic: str
    difficulty: str = "mixed"


class MarkQuizRequest(BaseModel):
    subject: str
    topic: str
    questions: List[str]
    student_answers: List[str]


def is_mock_mode():
    return os.getenv("MOCK_MODE", "false").lower() == "true"


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    return OpenAI(api_key=api_key)


def get_model():
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "ExamPrep Agent API is running",
        "mock_mode": is_mock_mode()
    }


@app.post("/study-plan")
def create_study_plan(profile: StudentProfile):
    try:
        weakest_topics = sorted(profile.topics, key=lambda topic: topic.confidence)
        weakest_topic_names = [topic.name for topic in weakest_topics[:3]]

        if is_mock_mode():
            return {
                "student_name": profile.student_name,
                "subject": profile.subject,
                "study_plan": f"""
Mock Study Plan for {profile.student_name}

Subject: {profile.subject}
Exam date: {profile.exam_date}
Goal grade: {profile.grade_goal}
Available study time: {profile.study_hours_per_day} hours per day

Priority topics:
{", ".join(weakest_topic_names)}

Suggested plan:
Day 1: Focus on {weakest_topic_names[0]}. Review notes and make a one-page summary.
Day 2: Practise 5 questions on {weakest_topic_names[0]}.
Day 3: Study {weakest_topic_names[1] if len(weakest_topic_names) > 1 else weakest_topic_names[0]} using examples.
Day 4: Do a mixed quiz across your weakest topics.
Day 5: Review mistakes and rewrite corrected answers.
Day 6: Timed practice session.
Day 7: Light revision and memory recall.

Exam strategy:
Start with your weakest topics first, use active recall, practise questions daily, and review mistakes carefully.
"""
            }

        client = get_client()

        topics_text = "\n".join(
            [f"- {topic.name}: confidence {topic.confidence}/5" for topic in profile.topics]
        )

        prompt = f"""
You are ExamPrep Agent, an encouraging AI study assistant.

Create a personalised study plan for this student.

Student name: {profile.student_name}
Subject: {profile.subject}
Exam date: {profile.exam_date}
Study hours per day: {profile.study_hours_per_day}
Goal grade: {profile.grade_goal}
Preferred study style: {profile.preferred_study_style}
Biggest worry: {profile.biggest_worry}

Topics and confidence:
{topics_text}

Rules:
- Prioritise weak topics first.
- Make the plan realistic.
- Use clear daily tasks.
- Include quiz practice.
- Give an exam strategy at the end.
"""

        response = client.responses.create(
            model=get_model(),
            input=prompt
        )

        return {
            "student_name": profile.student_name,
            "subject": profile.subject,
            "study_plan": response.output_text
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/quiz")
def create_quiz(request: QuizRequest):
    try:
        if is_mock_mode():
            return {
                "subject": request.subject,
                "topic": request.topic,
                "quiz": f"""
Mock Quiz: {request.topic}

Q1. What is the main idea of {request.topic}?
Expected answer: A clear explanation of the core concept.

Q2. Why is {request.topic} important in {request.subject}?
Expected answer: It helps solve or understand key problems in the subject.

Q3. Give one example of {request.topic}.
Expected answer: A relevant example connected to the topic.

Q4. What is a common mistake students make with {request.topic}?
Expected answer: Confusing the concept with a related but different idea.

Q5. How would you apply {request.topic} in an exam question?
Expected answer: Identify the problem, choose the correct method, and explain your reasoning.
"""
            }

        client = get_client()

        prompt = f"""
You are ExamPrep Agent.

Create a 5-question exam practice quiz.

Subject: {request.subject}
Topic: {request.topic}
Difficulty: {request.difficulty}

For each question:
- Write the question
- State the difficulty
- Give a short expected answer
"""

        response = client.responses.create(
            model=get_model(),
            input=prompt
        )

        return {
            "subject": request.subject,
            "topic": request.topic,
            "quiz": response.output_text
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/mark-quiz")
def mark_quiz(request: MarkQuizRequest):
    try:
        if is_mock_mode():
            feedback = ""

            for i, question in enumerate(request.questions):
                answer = request.student_answers[i] if i < len(request.student_answers) else "No answer given"
                feedback += f"""
Question {i + 1}: {question}
Student answer: {answer}
Score: 3/5
Feedback: This is a reasonable attempt. To improve, add more detail, use key terms, and connect your answer directly to the topic.
Corrected answer: A stronger answer should define the idea, explain why it matters, and give an example.
"""

            return {
                "subject": request.subject,
                "topic": request.topic,
                "feedback": feedback + """

Weak areas:
- Explanation depth
- Use of examples
- Exam-style structure

Next recommendation:
Revise the topic summary, then try another short quiz with more detailed answers.
"""
            }

        client = get_client()

        qa_text = ""
        for i, question in enumerate(request.questions):
            answer = request.student_answers[i] if i < len(request.student_answers) else "No answer given"
            qa_text += f"""
Question {i + 1}: {question}
Student answer: {answer}
"""

        prompt = f"""
You are ExamPrep Agent.

Mark the student's quiz answers.

Subject: {request.subject}
Topic: {request.topic}

Questions and answers:
{qa_text}

For each answer:
- Give a score out of 5
- Explain what was correct
- Explain what was missing
- Give a corrected answer

At the end:
- Identify weak areas
- Recommend what the student should study next
"""

        response = client.responses.create(
            model=get_model(),
            input=prompt
        )

        return {
            "subject": request.subject,
            "topic": request.topic,
            "feedback": response.output_text
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
@app.get("/student", response_class=HTMLResponse)
def student_page():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>ExamPrep Agent</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f3f4f6;
            color: #111827;
        }

        .header {
            background: #111827;
            color: white;
            padding: 28px;
            text-align: center;
        }

        .header h1 {
            margin: 0;
            font-size: 34px;
        }

        .header p {
            margin-top: 10px;
            color: #d1d5db;
        }

        .container {
            max-width: 1000px;
            margin: 30px auto;
            padding: 0 20px;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }

        .card {
            background: white;
            border-radius: 14px;
            padding: 24px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        }

        h2 {
            margin-top: 0;
            color: #1f2937;
        }

        label {
            display: block;
            margin-top: 14px;
            margin-bottom: 6px;
            font-weight: bold;
        }

        input, textarea {
            width: 100%;
            box-sizing: border-box;
            padding: 12px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            font-size: 15px;
        }

        textarea {
            resize: vertical;
        }

        .hint {
            font-size: 13px;
            color: #6b7280;
            margin-top: 6px;
        }

        button {
            margin-top: 18px;
            background: #2563eb;
            color: white;
            border: none;
            padding: 12px 18px;
            border-radius: 8px;
            font-size: 15px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        .result {
            margin-top: 18px;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 16px;
            white-space: pre-wrap;
            min-height: 80px;
        }

        .footer {
            text-align: center;
            color: #6b7280;
            font-size: 13px;
            margin: 30px 0;
        }

        @media (min-width: 900px) {
            .grid-two {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
        }
    </style>
</head>

<body>
    <div class="header">
        <h1>ExamPrep Agent</h1>
        <p>Plan your revision, practise with quizzes, and get feedback on your answers.</p>
    </div>

    <div class="container">
        <div class="card">
            <h2>1. Create your study plan</h2>
            <p>Enter your exam details. The agent will prioritise your weakest topics first.</p>

            <div class="grid-two">
                <div>
                    <label>Student name</label>
                    <input id="student_name" value="Alex">
                </div>

                <div>
                    <label>Subject</label>
                    <input id="subject" value="Machine Learning">
                </div>

                <div>
                    <label>Exam date</label>
                    <input id="exam_date" value="2026-06-01">
                </div>

                <div>
                    <label>Study hours per day</label>
                    <input id="study_hours_per_day" type="number" step="0.5" value="1.5">
                </div>

                <div>
                    <label>Goal grade</label>
                    <input id="grade_goal" value="Distinction">
                </div>

                <div>
                    <label>Preferred study style</label>
                    <input id="preferred_study_style" value="Practice questions">
                </div>
            </div>

            <label>Topics and confidence</label>
            <textarea id="topics" rows="5">Supervised Learning,4
Unsupervised Learning,2
Neural Networks,1</textarea>
            <div class="hint">Use one topic per line: Topic name, confidence from 1 to 5</div>

            <label>Biggest worry</label>
            <input id="biggest_worry" value="Neural networks">

            <button onclick="createStudyPlan()">Create Study Plan</button>

            <div id="result" class="result">Your study plan will appear here.</div>
        </div>

        <div class="grid-two">
            <div class="card">
                <h2>2. Generate a quiz</h2>
                <p>Choose a topic and the agent will generate practice questions.</p>

                <label>Quiz topic</label>
                <input id="quiz_topic" value="Neural Networks">

                <label>Quiz difficulty</label>
                <input id="quiz_difficulty" value="mixed">

                <button onclick="generateQuiz()">Generate Quiz</button>

                <div id="quiz_result" class="result">Your quiz will appear here.</div>
            </div>

            <div class="card">
                <h2>3. Mark your answers</h2>
                <p>Enter your answers and the agent will give feedback.</p>

                <label>Question 1</label>
                <input id="mark_q1" value="What is a neural network?">

                <label>Your answer 1</label>
                <textarea id="mark_a1" rows="3">A neural network is a model inspired by the brain.</textarea>

                <label>Question 2</label>
                <input id="mark_q2" value="What is overfitting?">

                <label>Your answer 2</label>
                <textarea id="mark_a2" rows="3">Overfitting is when a model memorises the training data.</textarea>

                <button onclick="markQuiz()">Mark Quiz</button>

                <div id="mark_result" class="result">Your feedback will appear here.</div>
            </div>
        </div>

        <div class="footer">
            ExamPrep Agent prototype | FastAPI + Azure App Service | Mock mode enabled
        </div>
    </div>

    <script>
        async function createStudyPlan() {
            const topicsText = document.getElementById("topics").value.trim();

            const topics = topicsText.split("\\n").map(function(line) {
                const parts = line.split(",");
                return {
                    name: parts[0].trim(),
                    confidence: Number(parts[1].trim())
                };
            });

            const payload = {
                student_name: document.getElementById("student_name").value,
                subject: document.getElementById("subject").value,
                exam_date: document.getElementById("exam_date").value,
                topics: topics,
                study_hours_per_day: Number(document.getElementById("study_hours_per_day").value),
                grade_goal: document.getElementById("grade_goal").value,
                preferred_study_style: document.getElementById("preferred_study_style").value,
                biggest_worry: document.getElementById("biggest_worry").value
            };

            document.getElementById("result").innerText = "Creating study plan...";

            const response = await fetch("/study-plan", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (response.ok) {
                document.getElementById("result").innerText = data.study_plan;
            } else {
                document.getElementById("result").innerText = "Error: " + JSON.stringify(data, null, 2);
            }
        }

        async function generateQuiz() {
            const payload = {
                subject: document.getElementById("subject").value,
                topic: document.getElementById("quiz_topic").value,
                difficulty: document.getElementById("quiz_difficulty").value
            };

            document.getElementById("quiz_result").innerText = "Generating quiz...";

            const response = await fetch("/quiz", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (response.ok) {
                document.getElementById("quiz_result").innerText = data.quiz;
            } else {
                document.getElementById("quiz_result").innerText = "Error: " + JSON.stringify(data, null, 2);
            }
        }

        async function markQuiz() {
            const payload = {
                subject: document.getElementById("subject").value,
                topic: document.getElementById("quiz_topic").value,
                questions: [
                    document.getElementById("mark_q1").value,
                    document.getElementById("mark_q2").value
                ],
                student_answers: [
                    document.getElementById("mark_a1").value,
                    document.getElementById("mark_a2").value
                ]
            };

            document.getElementById("mark_result").innerText = "Marking quiz...";

            const response = await fetch("/mark-quiz", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (response.ok) {
                document.getElementById("mark_result").innerText = data.feedback;
            } else {
                document.getElementById("mark_result").innerText = "Error: " + JSON.stringify(data, null, 2);
            }
        }
    </script>
</body>
</html>
"""
