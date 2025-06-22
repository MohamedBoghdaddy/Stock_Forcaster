import React, { useState, useEffect, useRef, useCallback } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  FaUser,
  FaRobot,
  FaChartLine,
  FaUserCircle,
  FaHome,
  FaCogs,
  FaChartBar,
  FaWallet,
  FaCalculator,
  FaBalanceScale,
} from "react-icons/fa";
import { BsPersonLinesFill } from "react-icons/bs";
import AvatarEditor from "react-avatar-editor";
import axios from "axios";
import { useAuthContext } from "../../../../context/AuthContext";
import "../../styles/Sidebar.css";

const Sidebar = () => {
  const { state } = useAuthContext();
  const { user, isAuthenticated } = state;
  const location = useLocation();

  const [profilePhoto, setProfilePhoto] = useState(user?.profilePhoto || null);
  const [image, setImage] = useState(null);
  const [scale, setScale] = useState(1.2);
  const [rotate, setRotate] = useState(0);
  const [isEditing, setIsEditing] = useState(false);
  const [error, setError] = useState("");
  const editorRef = useRef(null);

  useEffect(() => {
    const savedPhoto = localStorage.getItem("profilePhoto");
    if (savedPhoto) setProfilePhoto(savedPhoto);
    else if (user?.profilePhoto) setProfilePhoto(user.profilePhoto);
  }, [user]);

  const handleImageChange = (e) => {
    if (e.target.files?.[0]) {
      setImage(e.target.files[0]);
      setError("");
    }
  };

  const handleSave = async () => {
    if (!editorRef.current || !user?._id) return;
    try {
      const canvas = editorRef.current.getImageScaledToCanvas();
      const dataUrl = canvas.toDataURL();
      const blob = await fetch(dataUrl).then((res) => res.blob());

      const formData = new FormData();
      formData.append("photoFile", blob, "profile-photo.png");

      const res = await axios.put(
        `http://localhost:4000/api/users/update/${user._id}`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      const updatedPhoto = res.data.user.profilePhoto;
      setProfilePhoto(updatedPhoto);
      localStorage.setItem("profilePhoto", updatedPhoto);
      setIsEditing(false);
    } catch (err) {
      console.error(err);
      setError("Failed to upload profile photo.");
    }
  };

  const linkClass = (path) =>
    `flex items-center gap-2 p-2 rounded-lg transition ${
      location.pathname === path
        ? "bg-teal-600 text-white"
        : "text-white hover:text-teal-400"
    }`;

  const navItems = [
    { path: "/profile", label: "Profile", icon: <FaUser /> },
    {
      path: "/profile-card",
      label: "View Profile",
      icon: <BsPersonLinesFill />,
    },
    { path: "/statistics", label: "Statistics", icon: <FaChartBar /> },

    { path: "/chatbot", label: "AI Chatbot", icon: <FaRobot /> },
  ];

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2>
          {isAuthenticated && user ? `Hello, ${user.username}` : "Welcome"}
        </h2>
      </div>

      {isAuthenticated && (
        <div className="profile-photo-section">
          {profilePhoto ? (
            <img
              src={`http://localhost:4000/users/${profilePhoto}`}
              alt="Profile"
              className="profile-photo"
            />
          ) : (
            <FaUserCircle size={80} />
          )}

          <button
            className="mt-2 bg-gray-700 px-3 py-1 rounded text-sm"
            onClick={() => setIsEditing(!isEditing)}
          >
            Edit Photo
          </button>

          {isEditing && (
            <>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
              />
              {image && (
                <>
                  <AvatarEditor
                    ref={editorRef}
                    image={image}
                    width={150}
                    height={150}
                    border={20}
                    borderRadius={100}
                    scale={scale}
                    rotate={rotate}
                  />
                  <input
                    type="range"
                    min="1"
                    max="3"
                    step="0.1"
                    value={scale}
                    onChange={(e) => setScale(parseFloat(e.target.value))}
                  />
                  <div className="flex gap-2 mt-2">
                    <button onClick={() => setRotate((r) => r + 90)}>
                      Rotate
                    </button>
                    <button
                      className="bg-teal-600 text-white px-3 py-1 rounded"
                      onClick={handleSave}
                    >
                      Save
                    </button>
                    <button
                      className="bg-red-600 text-white px-3 py-1 rounded"
                      onClick={() => setIsEditing(false)}
                    >
                      Cancel
                    </button>
                  </div>
                </>
              )}
            </>
          )}
          {error && <p className="error-message">{error}</p>}
        </div>
      )}

      <ul className="sidebar-menu">
        {navItems.map((item) => (
          <li key={item.path}>
            <Link to={item.path} className={linkClass(item.path)}>
              {item.icon} {item.label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default React.memo(Sidebar);
