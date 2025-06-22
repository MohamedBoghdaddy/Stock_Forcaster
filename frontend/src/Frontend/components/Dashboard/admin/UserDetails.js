import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import "../../styles/admin.css";
import { toast } from "react-toastify";

const UserDetails = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const [user, setUser] = useState(null);
    const [newRole, setNewRole] = useState("");
    const [chatLogs, setChatLogs] = useState([]);
    const [questionnaireLogs, setQuestionnaireLogs] = useState([]);

    useEffect(() => {
        const fetchUser = async () => {
            try {
                const res = await axios.get(`http://localhost:4000/api/users/${id}`);
                setUser(res.data);
                setNewRole(res.data.role);
            } catch (err) {
                console.error("Failed to fetch user", err);
            }
        };

        const fetchLogs = async () => {
            try {
                const chatRes = await axios.get(`http://localhost:4000/api/chat/user/${id}`);
                const questionnaireRes = await axios.get(`http://localhost:4000/api/questionnaire/user/${id}`);
                setChatLogs(Array.isArray(chatRes.data) ? chatRes.data : []);
                setQuestionnaireLogs(Array.isArray(questionnaireRes.data) ? questionnaireRes.data : []);
            } catch (err) {
                console.error("Failed to fetch logs", err);
            }
        };

        fetchUser();
        fetchLogs();
    }, [id]);

    const handleSave = async () => {
        if (!window.confirm("Are you sure you want to update this user's role?")) return;

        try {
            await axios.put(`http://localhost:4000/api/admin/update-role/${id}`, {
                newRole,
            });
            toast.success("Role updated successfully");
        } catch (err) {
            console.error(err);
            toast.error("Failed to update role");
        }
    };


    if (!user) return <div>Loading...</div>;

    return (
        <div className="admin-dashboard">
            <h1>User Details</h1>
            <div className="user-details-box">
                <p><strong>Username:</strong> {user.username}</p>
                <p><strong>Email:</strong> {user.email}</p>

                <p><strong>Role:</strong> {user.role}</p>

                <p><strong>Status:</strong>
                    <span className={`status-badge ${user.blocked ? "blocked" : "active"}`}>
                        {user.blocked ? "Blocked" : "Active"}
                    </span>
                </p>

                {/* <p><strong>Joined:</strong> {user.createdAt ? new Date(user.createdAt).toLocaleString() : "N/A"}</p>
<p><strong>Last Login:</strong> {user.lastLogin ? new Date(user.lastLogin).toLocaleString() : "N/A"}</p>
<p><strong>Last IP:</strong> {user.lastIP && user.lastIP !== "::1" ? user.lastIP : "N/A"}</p> */}




                <button onClick={handleSave} className="save-btn">Save Changes</button>
                <button onClick={() => navigate("/admin/dashboard")} className="back-btn">
                    🔙 Back to Dashboard
                </button>
            </div>

            <h2>Activity Log</h2>
            {Array.isArray(user.activityLog) && user.activityLog.length > 0 ? (
                <ul className="activity-log">
                    {user.activityLog.map((log, idx) => (
                        <li key={idx}>
                            <strong>{log.action}</strong> - {new Date(log.timestamp).toLocaleString()}
                        </li>
                    ))}
                </ul>
            ) : (
                <p>No activity recorded.</p>
            )}

            <h2>AI Chats</h2>
            {Array.isArray(chatLogs) && chatLogs.length > 0 ? (
                <ul>
                    {chatLogs.map((c, idx) => (
                        <li key={idx}>{c.message}</li>
                    ))}
                </ul>
            ) : <p>No AI chats found.</p>}


            <h2>Questionnaires</h2>
            {Array.isArray(questionnaireLogs) && questionnaireLogs.length > 0 ? (
                <ul>
                    {questionnaireLogs.map((q, idx) => (
                        <li key={idx}>{q.summary || "Completed form"}</li>
                    ))}
                </ul>
            ) : <p>No questionnaires found.</p>}

        </div>
    );
};

export default UserDetails;
