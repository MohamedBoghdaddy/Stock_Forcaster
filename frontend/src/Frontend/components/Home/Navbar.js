import React, { useState, useEffect, useRef } from "react";
import {
  Navbar,
  Nav,
  Container,
  Modal,
  Form,
  Row,
  Col,
  Button,
} from "react-bootstrap";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faUser, faSignOutAlt } from "@fortawesome/free-solid-svg-icons";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { Link as ScrollLink } from "react-scroll";
import { useAuthContext } from "../../../context/AuthContext";
import { useLogout } from "../../../hooks/useLogout.js";
import Login from "../LOGIN&REGISTRATION/Login/Login"; // adjust path if needed
import Logo from "../../../assets/latest_logo.png"; // or second logo path
import "../styles/navbar.css";
import "bootstrap/dist/css/bootstrap.min.css";

const NavBar = () => {
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const { state } = useAuthContext();
  const { user, isAuthenticated } = state;
  const { logout } = useLogout();
  const navigate = useNavigate();
  const location = useLocation();

  const navRef = useRef();

  // Scroll effect to add background class
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Close mobile menu on route change
  useEffect(() => {
    setExpanded(false);
  }, [location.pathname]);

  // Click outside to close mobile menu
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (expanded && navRef.current && !navRef.current.contains(e.target)) {
        setExpanded(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [expanded]);

  // Prevent scrolling when menu open
  useEffect(() => {
    if (expanded) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [expanded]);

  const handleLoginModalOpen = () => setShowLoginModal(true);
  const handleLoginModalClose = () => setShowLoginModal(false);

  const handleLogout = async () => {
    try {
      await logout();
      setExpanded(false);
      navigate("/");
    } catch (err) {
      console.error("Logout failed:", err);
    }
  };

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (searchTerm.trim()) {
      navigate(`/workspaces?term=${encodeURIComponent(searchTerm.trim())}`);
      setSearchTerm("");
      setExpanded(false);
    }
  };

  return (
    <>
      <Navbar
        expand="lg"
        className={`navbar ${scrolled ? "navbar-scrolled" : ""}`}
        variant="dark"
        expanded={expanded}
        ref={navRef}
      >
        <Container fluid>
          <Navbar.Brand as={Link} to="/" className="navbar-logo">
            <img
              src={Logo}
              alt="Financial AI Advisor"
              style={{ width: "80px", height: "57px" }}
            />
          </Navbar.Brand>
          <Navbar.Toggle
            aria-controls="basic-navbar-nav"
            onClick={() => setExpanded(!expanded)}
            className="navbar-toggler"
          />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="ms-auto navbar-nav" navbarScroll>
              {/* Smooth scroll links */}
              <ScrollLink
                to="hero-section"
                smooth
                className="nav-link"
                onClick={() => setExpanded(false)}
                tabIndex={0}
              >
                Home
              </ScrollLink>
              <ScrollLink
                to="WhoWeAre"
                smooth
                className="nav-link"
                onClick={() => setExpanded(false)}
                tabIndex={0}
              >
                About
              </ScrollLink>
             
              <Nav.Link
                as={Link}
                to="/contact"
                className={`nav-link ${
                  location.pathname === "/contact" ? "active" : ""
                }`}
                onClick={() => setExpanded(false)}
              >
                Contact
              </Nav.Link>
              {isAuthenticated && user && (
                <Nav.Link
                  as={Link}
                  to="/dashboard"
                  className={`nav-link ${
                    location.pathname === "/dashboard" ? "active" : ""
                  }`}
                  onClick={() => setExpanded(false)}
                >
                  Dashboard
                </Nav.Link>
              )}

              {/* Auth Buttons */}
              {isAuthenticated && user ? (
                <Nav.Link
                  role="button"
                  tabIndex={0}
                  onClick={handleLogout}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") handleLogout();
                  }}
                  className="nav-link logout-link"
                >
                  <FontAwesomeIcon icon={faSignOutAlt} /> Logout
                </Nav.Link>
              ) : (
                <Nav.Link
                  role="button"
                  tabIndex={0}
                  onClick={() => {
                    handleLoginModalOpen();
                    setExpanded(false);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ")
                      handleLoginModalOpen();
                  }}
                  className="nav-link login-link"
                >
                  <FontAwesomeIcon icon={faUser} /> Login
                </Nav.Link>
              )}

              {/* Search Form */}
              <Form
                className="d-flex align-items-center mt-2 mt-lg-0"
                onSubmit={handleSearchSubmit}
                role="search"
                aria-label="Site search"
              >
                <Row className="g-0 align-items-center">
                  <Col xs="auto">
                    <Form.Control
                      type="search"
                      placeholder="Search"
                      aria-label="Search"
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="me-2"
                    />
                  </Col>
                  <Col xs="auto">
                    <Button type="submit" variant="outline-light" size="sm">
                      Submit
                    </Button>
                  </Col>
                </Row>
              </Form>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      {/* Login Modal */}
      <Modal show={showLoginModal} onHide={handleLoginModalClose} centered>
        <Modal.Header closeButton />
        <Modal.Body>
          <Login onLoginSuccess={handleLoginModalClose} />
        </Modal.Body>
      </Modal>
    </>
  );
};

export default NavBar;
