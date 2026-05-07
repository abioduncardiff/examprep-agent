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
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 850px;
            margin: 40px auto;
            padding: 20px;
            background-color: #f7f7f7;
        }

        h1 {
            color: #222;
        }

        label {
            display: block;
            margin-top: 15px;
            font-weight: bold;
        }

        input, textarea, select {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            font-size: 16px;
        }

        button {
            margin-top: 20px;
            padding: 12px 18px;
            font-size: 16px;
            cursor: pointer;
        }

        #result {
            margin-top: 30px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <h1>ExamPrep Agent</h1>
    <p>Enter your exam details and the agent will create a study plan.</p>

    <label>Student name</label>
    <input id="student_name" value="Alex">

    <label>Subject</label>
    <input id="subject" value="Machine Learning">

    <label>Exam date</label>
    <input id="exam_date" value="2026-06-01">

    <label>Topics</label>
    <textarea id="topics" rows="5">Supervised Learning,4
Unsupervised Learning,2
Neural Networks,1</textarea>
    <p>Write one topic per line like this: Topic name, confidence from 1 to 5</p>

    <label>Study hours per day</label>
    <input id="study_hours_per_day" type="number" step="0.5" value="1.5">

    <label>Goal grade</label>
    <input id="grade_goal" value="Distinction">

    <label>Preferred study style</label>
    <input id="preferred_study_style" value="Practice questions">

    <label>Biggest worry</label>
    <input id="biggest_worry" value="Neural networks">

    <button onclick="createStudyPlan()">Create Study Plan</button>
<hr>

<h2>Generate a Quiz</h2>

<label>Quiz topic</label><br>
<input id="quiz_topic" value="Neural Networks"><br><br>

<label>Quiz difficulty</label><br>
<input id="quiz_difficulty" value="mixed"><br><br>

<button onclick="generateQuiz()">Generate Quiz</button>

<pre id="quiz_result">Your quiz will appear here.</pre>
<hr>

<h2>Mark Quiz Answers</h2>

<label>Question 1</label><br>
<input id="mark_q1" value="What is a neural network?"><br><br>

<label>Your answer 1</label><br>
<textarea id="mark_a1" rows="3" cols="60">A neural network is a model inspired by the brain.</textarea><br><br>

<label>Question 2</label><br>
<input id="mark_q2" value="What is overfitting?"><br><br>

<label>Your answer 2</label><br>
<textarea id="mark_a2" rows="3" cols="60">Overfitting is when a model memorises the training data.</textarea><br><br>

<button onclick="markQuiz()">Mark Quiz</button>

<pre id="mark_result">Your feedback will appear here.</pre>
    <div id="result">Your study plan will appear here.</div>
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
    <script>
        async function createStudyPlan() {
            const topicsText = document.getElementById("topics").value.trim();

            const topics = topicsText.split("\\n").map(line => {
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
    </script>
</body>
</html>
"""
