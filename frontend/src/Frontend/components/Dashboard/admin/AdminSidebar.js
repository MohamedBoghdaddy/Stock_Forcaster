import React from "react";
import { Link } from "react-router-dom";
import "../../styles/admin.css";

const AdminSidebar = () => {
  return (
    <div className="admin-sidebar">
      <h2 className="sidebar-title">Admin Panel</h2>
      <ul>
        <li><Link to="/admin/dashboard">Dashboard</Link></li>
        <li><Link to="/admin/users">Users</Link></li>
        <li><Link to="/admin/chatlogs">AI Chats</Link></li>
        <li><Link to="/admin/questionnaires">Questionnaires</Link></li>
        <li><Link to="/">Back to Website</Link></li>
      </ul>
    </div>
  );
};

export default AdminSidebar;
