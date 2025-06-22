import React, { useState } from "react";
import { Navbar, Nav, Container, Modal } from "react-bootstrap";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faUser, faSignOutAlt } from "@fortawesome/free-solid-svg-icons";
import { Link, useNavigate } from "react-router-dom";
import { Link as ScrollLink } from "react-scroll";
import Logo from "../../../assets/latest_logo.png";
import Login from "../LOGIN&REGISTRATION/Login/Login"; // Adjust path to Login component
import { useAuthContext } from "../../../context/AuthContext";
import { useLogout } from "../../../hooks/useLogout.js";
import "../styles/navbar.css";
import "bootstrap/dist/css/bootstrap.min.css";

const NavBar = () => {
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const navigate = useNavigate();
  const { state } = useAuthContext();

  const { user, isAuthenticated } = state; // Use isAuthenticated
  const { logout } = useLogout();

  const handleLoginModalOpen = () => {
    setShowLoginModal(true);
  };

  const handleLoginModalClose = () => {
    setShowLoginModal(false);
  };

  const handleNavCollapse = () => setExpanded(!expanded);

  const handleLogout = async () => {
    logout();
    navigate("/");
  };

  return (
    <Navbar expand="lg" className="navbar" variant="dark" expanded={expanded}>
      <Container fluid>
        <Navbar.Brand as={Link} to="/" className="navbar-brand">
          <img
            src={Logo}
            alt="Company Logo"
            style={{ width: "80px", height: "57px", top: 0 }}
          />
        </Navbar.Brand>
        <Navbar.Toggle
          aria-controls="basic-navbar-nav"
          className="navbar-toggler"
          onClick={handleNavCollapse}
        />
        <Navbar.Collapse id="navbarScroll" className="navbar-collapse">
          <Nav className="navbar-nav ms-auto" navbarScroll>
            <Link
              to="/"
              smooth
              className="nav-link"
              onClick={handleNavCollapse}
            >
              Home
            </Link>
            <ScrollLink
              to="/Dashboard"
              className="nav-link"
              onClick={handleNavCollapse}
            >
              Dashboard{" "}
            </ScrollLink>
            <ScrollLink
              to="/contact"
              className="nav-link"
              onClick={handleNavCollapse}
            >
              Contact Us
            </ScrollLink>

            {isAuthenticated && user ? (
              <div
                className="nav-link"
                role="button"
                tabIndex="0"
                onClick={handleLogout}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    handleLogout();
                  }
                }}
              >
                <FontAwesomeIcon icon={faSignOutAlt} /> Logout
              </div>
            ) : (
              <div
                className="nav-link"
                role="button"
                tabIndex="0"
                onClick={() => {
                  handleLoginModalOpen();
                  handleNavCollapse();
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    handleLoginModalOpen();
                    handleNavCollapse();
                  }
                }}
              >
                <FontAwesomeIcon icon={faUser} />
              </div>
            )}
          </Nav>
        </Navbar.Collapse>
      </Container>

      <Modal show={showLoginModal} onHide={handleLoginModalClose} centered>
        <Modal.Header closeButton></Modal.Header>
        <Modal.Body>
          <Login onLoginSuccess={handleLoginModalClose} /> {/* Pass callback */}
        </Modal.Body>
      </Modal>
    </Navbar>
  );
};

export default NavBar;
