import React, { useState, useEffect, useContext, useCallback } from "react";
import PropTypes from "prop-types";
import { useAuthContext } from "../../../../context/AuthContext";
import { DashboardContext } from "../../../../context/DashboardContext";
import { BsPersonCircle } from "react-icons/bs";
import { toast } from "react-toastify";
import "../../styles/Profile.css";

const questions = [
  { id: "age", text: "What's your age?", type: "number", min: 18, max: 120 },
  {
    id: "employmentStatus",
    text: "What's your employment status?",
    type: "select",
    options: ["Employed", "Self-employed", "Unemployed", "Student", "Retired"],
  },
  { id: "salary", text: "Your Salary?", type: "number", min: 0 },
  {
    id: "homeOwnership",
    text: "Do you own or rent your home?",
    type: "select",
    options: ["Own", "Rent", "Other"],
  },
  {
    id: "hasDebt",
    text: "Do you currently have any debts?",
    type: "select",
    options: ["Yes", "No"],
  },
  {
    id: "lifestyle",
    text: "What type of lifestyle best describes you?",
    type: "select",
    options: [
      { label: "Minimalist (low spending, high saving)", value: "Minimalist" },
      { label: "Balanced (moderate spending & saving)", value: "Balanced" },
      { label: "Spender (high spending, lower saving)", value: "Spender" },
    ],
  },
  {
    id: "riskTolerance",
    text: "How comfortable are you with unpredictable situations?",
    type: "slider",
    min: 1,
    max: 10,
    labels: ["Very Uncomfortable", "Very Comfortable"],
  },
  {
    id: "investmentApproach",
    text: "How do you usually handle a surplus of money?",
    type: "slider",
    min: 1,
    max: 10,
    labels: ["Spend It", "Invest It"],
  },
  {
    id: "emergencyPreparedness",
    text: "If a major unexpected expense arises, how prepared do you feel?",
    type: "slider",
    min: 1,
    max: 10,
    labels: ["Not Prepared", "Very Prepared"],
  },
  {
    id: "financialTracking",
    text: "How often do you research financial trends?",
    type: "slider",
    min: 1,
    max: 10,
    labels: ["Never", "Daily"],
  },
  {
    id: "futureSecurity",
    text: "How much do you prioritize future financial security over present comfort?",
    type: "slider",
    min: 1,
    max: 10,
    labels: ["Present Comfort", "Future Security"],
  },
  {
    id: "spendingDiscipline",
    text: "How easily do you say 'no' to unplanned purchases?",
    type: "slider",
    min: 1,
    max: 10,
    labels: ["Very Difficult", "Very Easy"],
  },
  {
    id: "assetAllocation",
    text: "If given a large sum of money today, how much would you allocate toward long-term assets?",
    type: "slider",
    min: 1,
    max: 10,
    labels: ["None", "All"],
  },
  {
    id: "riskTaking",
    text: "When it comes to financial risks, where do you stand?",
    type: "slider",
    min: 1,
    max: 10,
    labels: ["Risk Averse", "Risk Seeking"],
  },
  {
    id: "dependents",
    text: "Do you have dependents (children, elderly, etc.)?",
    type: "select",
    options: ["Yes", "No"],
  },
  {
    id: "financialGoals",
    text: "Briefly describe your primary financial goals:",
    type: "textarea",
    placeholder: "E.g., Save for retirement, buy a home, pay off debt...",
  },
];

const Profile = () => {
  const { state: authState } = useAuthContext();
  const { user } = authState || {};
  const {
    state: dashState = {},
    actions: { submitProfile, fetchProfile } = {},
    loading: { profile: profileLoading } = {},
    aiAdvice = null,
    goalPlan = null,
  } = useContext(DashboardContext);

  const [formData, setFormData] = useState({});
  const [step, setStep] = useState(0);
  const [editMode, setEditMode] = useState(true); // Changed to true to show questions first
  const [validationErrors, setValidationErrors] = useState({});
  const [localLoading, setLocalLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  // Custom expense handlers
  const addCustomExpense = useCallback(() => {
    setFormData((prev) => ({
      ...prev,
      customExpenses: [...(prev.customExpenses || []), { name: "", amount: "" }],
    }));
  }, []);

  const removeCustomExpense = useCallback((index) => {
    setFormData((prev) => ({
      ...prev,
      customExpenses: (prev.customExpenses || []).filter((_, i) => i !== index),
    }));
  }, []);

  const updateCustomExpense = useCallback((index, key, value) => {
    setFormData((prev) => {
      const updated = [...(prev.customExpenses || [])];
      updated[index] = { ...updated[index], [key]: value };
      return { ...prev, customExpenses: updated };
    });
  }, []);

  const validateField = useCallback((id, value) => {
    const question = questions.find((q) => q.id === id);
    if (!question) return "";
    if (question.type === "number") {
      const numValue = Number(value);
      if (value === "") return "This field is required";
      if (question.min !== undefined && numValue < question.min)
        return `Minimum value is ${question.min}`;
      if (question.max !== undefined && numValue > question.max)
        return `Maximum value is ${question.max}`;
    }
    if (question.type === "select" && !value) return "Please select an option";
    return "";
  }, []);

  const handleChange = useCallback(
    (id, value) => {
      const error = validateField(id, value);
      setValidationErrors((prev) => ({ ...prev, [id]: error }));
      setFormData((prev) => ({ ...prev, [id]: value }));
    },
    [validateField]
  );

  const renderQuestionInput = useCallback(
    (question) => {
      const value = formData[question.id] ?? "";
      const error = validationErrors[question.id];

      switch (question.type) {
        case "number":
          return (
            <div className="input-group">
              <input
                type="number"
                value={value}
                onChange={(e) => handleChange(question.id, e.target.value)}
                min={question.min}
                max={question.max}
                className="input-field"
              />
              {error && <span className="error-text">{error}</span>}
            </div>
          );
        case "select":
          return (
            <div className="input-group">
              <select
                value={value}
                onChange={(e) => handleChange(question.id, e.target.value)}
                className="input-field"
              >
                <option value="">Select an option</option>
                {question.options.map((opt) => (
                  <option key={opt.value || opt} value={opt.value || opt}>
                    {opt.label || opt}
                  </option>
                ))}
              </select>
              {error && <span className="error-text">{error}</span>}
            </div>
          );
        case "slider":
          return (
            <div className="input-group slider-container">
              <div className="slider-wrapper">
                <input
                  type="range"
                  min={question.min}
                  max={question.max}
                  value={value || 5}
                  onChange={(e) => handleChange(question.id, e.target.value)}
                  className="slider"
                />
                <div className="slider-labels">
                  <span>{question.labels[0]}</span>
                  <span className="slider-value">{value || 5}</span>
                  <span>{question.labels[1]}</span>
                </div>
              </div>
            </div>
          );
        case "textarea":
          return (
            <div className="input-group">
              <textarea
                value={value}
                onChange={(e) => handleChange(question.id, e.target.value)}
                className="input-field"
                placeholder={question.placeholder}
                rows={4}
              />
              {error && <span className="error-text">{error}</span>}
            </div>
          );
        default:
          return null;
      }
    },
    [formData, validationErrors, handleChange]
  );

  const handleNext = useCallback(() => {
    const currentQuestion = questions[step];
    const error = validateField(currentQuestion.id, formData[currentQuestion.id]);
    if (error) {
      setValidationErrors((prev) => ({ ...prev, [currentQuestion.id]: error }));
      toast.error(`Please complete the current question: ${error}`);
      return;
    }
    setStep((prev) => Math.min(prev + 1, questions.length - 1));
  }, [step, formData, validateField]);

  const handleBack = useCallback(() => setStep((prev) => Math.max(prev - 1, 0)), []);

  const handleSubmit = useCallback(async () => {
    const currentQuestion = questions[step];
    const error = validateField(currentQuestion.id, formData[currentQuestion.id]);
    if (error) {
      setValidationErrors((prev) => ({ ...prev, [currentQuestion.id]: error }));
      toast.error(`Please complete the current question: ${error}`);
      return;
    }
    setSubmitting(true);
    try {
      await submitProfile({ ...formData });
      setEditMode(false);
    } catch (error) {
      console.error("Profile save error:", error);
      const message = error.response?.data?.message || error.message;
      toast.error(message || "Failed to save profile");
    } finally {
      setSubmitting(false);
    }
  }, [step, formData, validateField, submitProfile]);

  useEffect(() => {
    const controller = new AbortController();
    let isMounted = true;
    const loadProfile = async () => {
      if (!user?.token) {
        setLocalLoading(false);
        return;
      }
      try {
        await fetchProfile();
      } catch (err) {
        if (isMounted) {
          console.error("Profile load error:", err);
          toast.error(err.message || "Failed to load profile data");
        }
      } finally {
        if (isMounted) setLocalLoading(false);
      }
    };
    loadProfile();
    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [user, fetchProfile]);

  useEffect(() => {
    if (dashState.profile) {
      setFormData(dashState.profile);
    }
  }, [dashState.profile]);

  if (localLoading || profileLoading) {
    return (
      <div className="profile-container">
        <div className="loader-container">
          <div className="loader"></div>
          <p>Loading your profile...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="profile-container">
      <h2>📋 My Financial Profile</h2>
      <div className="profile-card">
        <BsPersonCircle className="profile-icon" />
     {editMode ? (
  <QuestionFlow
    step={step}
    questions={questions}
    formData={formData}
    validationErrors={validationErrors}
    handleBack={handleBack}
    handleNext={handleNext}
    handleSubmit={handleSubmit}
    submitting={submitting}
    updateCustomExpense={updateCustomExpense}
    removeCustomExpense={removeCustomExpense}
    addCustomExpense={addCustomExpense}
    renderQuestionInput={renderQuestionInput}
  />
) : (
  <div className="submitted-success-message">
    <h3>✅ Profile Saved Successfully!</h3>
    <p>
      You can now view your full profile details from the{" "}
      <strong>“View Profile”</strong> page.
    </p>
  </div>
)}

      </div>
    </div>
  );
};

const ProfileView = ({ formData, questions, aiAdvice, goalPlan, onEditAll, onEditSpecific }) => (
  <div className="submitted-results">
    <h3>📌 Your Responses</h3>
    <div className="profile-summary">
      {questions.map((q, index) => (
        <div key={q.id} className="profile-item">
          <p>
            <strong>{q.text}:</strong>{" "}
            {formData[q.id]?.toString() || "Not answered"}
          </p>
          <button onClick={() => onEditSpecific(index)} className="edit-btn">
            ✏️ Edit
          </button>
        </div>
      ))}
    </div>
    <button onClick={onEditAll} className="edit-all-btn">
      📝 Edit All Responses
    </button>

    {aiAdvice && (
      <div className="ai-advice-section">
        <h3>💡 AI Financial Recommendations</h3>
        <div className="advice-summary">
          <p>{aiAdvice.summary}</p>
          <ul>
            {aiAdvice.advice?.map((tip) => (
              <li key={tip}>✅ {tip}</li>
            ))}
          </ul>
        </div>
      </div>
    )}

    {goalPlan && (
      <div className="goal-plan-section">
        <h3>🎯 Your Personalized Goal Plan</h3>
        {goalPlan.map((g) => (
          <div key={g.goal} className="goal-item">
            <h4>🏁 {g.goal}</h4>
            <ul>
              {g.milestones?.map((m) => (
                <li key={`${g.goal}-${m.task}`}>
                  📅 {m.task} by <strong>{m.target_date}</strong>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    )}
  </div>
);

const QuestionFlow = ({
  step,
  questions,
  formData,
  validationErrors,
  handleBack,
  handleNext,
  handleSubmit,
  submitting,
  updateCustomExpense,
  removeCustomExpense,
  addCustomExpense,
  renderQuestionInput,
}) => (
  <div className="question-block">
    <div className="progress-tracker">
      <div
        className="progress-bar"
        style={{ width: `${(step / (questions.length - 1)) * 100}%` }}
      ></div>
      <span>
        Question {step + 1} of {questions.length}
      </span>
    </div>

    <div className="question-content">
      <h3>{questions[step].text}</h3>
      {renderQuestionInput(questions[step])}

      {step === questions.length - 1 && (
        <div className="custom-expense-section">
          <h4>➕ Additional Monthly Expenses</h4>
          {(formData.customExpenses || []).map((item, index) => (
            <div key={`expense-${index}`} className="expense-row">
              <input
                type="text"
                placeholder="Expense name"
                value={item.name}
                onChange={(e) =>
                  updateCustomExpense(index, "name", e.target.value)
                }
              />
              <input
                type="number"
                placeholder="Amount ($)"
                value={item.amount}
                onChange={(e) =>
                  updateCustomExpense(index, "amount", e.target.value)
                }
              />
              <button
                onClick={() => removeCustomExpense(index)}
                className="remove-btn"
              >
                🗑️
              </button>
            </div>
          ))}
          <button onClick={addCustomExpense} className="add-expense-btn">
            ➕ Add Another Expense
          </button>
        </div>
      )}
    </div>

    <div className="navigation-buttons">
      {step > 0 && (
        <button onClick={handleBack} className="nav-btn back">
          ⬅️ Previous
        </button>
      )}
      {step < questions.length - 1 ? (
        <button onClick={handleNext} className="nav-btn next">
          Next Question ➡️
        </button>
      ) : (
        <button
          onClick={handleSubmit}
          className="submit-btn"
          disabled={submitting}
        >
          {submitting ? "Saving..." : "Save Profile ✅"}
        </button>
      )}
    </div>
  </div>
);

export default Profile;