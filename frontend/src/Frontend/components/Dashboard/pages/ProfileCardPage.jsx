import React, { useEffect, useState, useContext } from "react";
import { DashboardContext } from "../../../../context/DashboardContext";
import { BsPersonCircle } from "react-icons/bs";
import "../../styles/viewprofile.css";
import { toast } from "react-toastify";


// ⚙️ Reference question metadata for field types/options only
const questionMeta = {
  age: { type: "number", min: 18, max: 120 },
  employmentStatus: {
    type: "select",
    options: ["Employed", "Self-employed", "Unemployed", "Student", "Retired"],
  },
  salary: { type: "number", min: 0 },
  homeOwnership: {
    type: "select",
    options: ["Own", "Rent", "Other"],
  },
  hasDebt: {
    type: "select",
    options: ["Yes", "No"],
  },
  lifestyle: {
    type: "select",
    options: [
      { label: "Minimalist (low spending, high saving)", value: "Minimalist" },
      { label: "Balanced (moderate spending & saving)", value: "Balanced" },
      { label: "Spender (high spending, lower saving)", value: "Spender" },
    ],
  },
  riskTolerance: { type: "slider", min: 1, max: 10 },
  investmentApproach: { type: "slider", min: 1, max: 10 },
  emergencyPreparedness: { type: "slider", min: 1, max: 10 },
  financialTracking: { type: "slider", min: 1, max: 10 },
  futureSecurity: { type: "slider", min: 1, max: 10 },
  spendingDiscipline: { type: "slider", min: 1, max: 10 },
  assetAllocation: { type: "slider", min: 1, max: 10 },
  riskTaking: { type: "slider", min: 1, max: 10 },
  dependents: {
    type: "select",
    options: ["Yes", "No"],
  },
  financialGoals: { type: "textarea" },
  customExpenses: { type: "text" },
  totalMonthlyExpenses: { type: "number" },
};

const ProfileCardPage = () => {
  const dashboardContext = useContext(DashboardContext);
  const dashState = dashboardContext || {};
  const fetchProfile = dashboardContext?.actions?.fetchProfile;
  const submitProfile = dashboardContext?.actions?.submitProfile;
  const profileLoading = dashboardContext?.loading?.profile || false;

  const [localLoading, setLocalLoading] = useState(true);
  const [editField, setEditField] = useState(null);
  const [editedValues, setEditedValues] = useState({});
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        if (fetchProfile) {
          await fetchProfile();
        } else {
          toast.error("Profile fetch function is missing");
        }
      } catch (err) {
        toast.error(err.message || "Error loading profile.");
      } finally {
        setLocalLoading(false);
      }
    };
    load();
  }, [fetchProfile]);

  const profileData = dashState.profile;

  const displayableKeys = Object.keys(questionMeta);

  const handleEditClick = (key) => {
    setEditField(key);
    setEditedValues((prev) => ({
      ...prev,
      [key]: profileData[key],
    }));
  };

  const handleChange = (key, value) => {
    setEditedValues((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleSave = async () => {
    if (!editField) return;
    setSubmitting(true);
    try {
      const payload = { ...profileData, ...editedValues };
      await submitProfile(payload);
      toast.success("✅ Profile updated!");
      setEditField(null);
      await fetchProfile();
    } catch (err) {
      toast.error("❌ Failed to update profile");
    } finally {
      setSubmitting(false);
    }
  };

  if (localLoading || profileLoading) {
    return (
      <div className="profile-container">
        <p>Loading profile...</p>
      </div>
    );
  }

  if (!profileData || Object.keys(profileData).length === 0) {
    return (
      <div className="profile-container">
        <p>No profile data available.</p>
      </div>
    );
  }

  return (
    <div className="profile-container">
      <h2>👤 User Profile Overview</h2>
      <div className="profile-card">
        <BsPersonCircle className="profile-icon" />
        <div className="profile-summary">
          {displayableKeys.map((key) => {
            const meta = questionMeta[key];
            const value = profileData[key];

            return (
              <div key={key} className="profile-item">
                <strong>{formatKey(key)}</strong>
                {editField === key ? (
                  meta.type === "select" ? (
                    <select
                      className="edit-input"
                      value={editedValues[key] ?? ""}
                      onChange={(e) => handleChange(key, e.target.value)}
                    >
                      <option value="">Select...</option>
                      {(meta.options || []).map((opt) =>
                        typeof opt === "string" ? (
                          <option key={opt} value={opt}>
                            {opt}
                          </option>
                        ) : (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        )
                      )}
                    </select>
                  ) : meta.type === "slider" || meta.type === "number" ? (
                    <input
                      type="number"
                      className="edit-input"
                      value={editedValues[key] ?? ""}
                      onChange={(e) => handleChange(key, e.target.value)}
                      min={meta.min}
                      max={meta.max}
                    />
                  ) : meta.type === "textarea" ? (
                    <textarea
                      className="edit-input"
                      rows={3}
                      value={editedValues[key] ?? ""}
                      onChange={(e) => handleChange(key, e.target.value)}
                    />
                  ) : (
                    <input
                      className="edit-input"
                      type="text"
                      value={editedValues[key] ?? ""}
                      onChange={(e) => handleChange(key, e.target.value)}
                    />
                  )
                ) : (
                  <div className="profile-value-row" onClick={() => handleEditClick(key)}>
  <span>{String(value)}</span>
</div>

                )}
              </div>
            );
          })}
        </div>

        {editField && (
          <button className="save-btn" onClick={handleSave} disabled={submitting}>
            {submitting ? "Saving..." : "💾 Save Changes"}
          </button>
        )}
      </div>
    </div>
  );
};

const formatKey = (key) =>
  key
    .replace(/([A-Z])/g, " $1")
    .replace(/^./, (str) => str.toUpperCase())
    .replace(/_/g, " ");

export default ProfileCardPage;
