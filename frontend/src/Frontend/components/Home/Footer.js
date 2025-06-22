import React from "react";
import { Container, Row, Col, Form, Button } from "react-bootstrap";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faFacebookF,
  faTwitter,
  faLinkedinIn,
  faInstagram,
} from "@fortawesome/free-brands-svg-icons";
import OurLogo from "../../../assets/latest_logo.png";

import "../styles/footer.css";

function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="footer">
      <Container>
        <Row className="footer-content">
          <Col lg={4} md={6} className="mb-4 mb-md-0">
            <div className="footer-brand">
              <img
                src={OurLogo}
                alt="Financial AI Advisor"
                className="footer-logo"
              />
              <p className="mt-3">
                Manage your finances like never before. Our AI-powered platform
                provides personalized budgeting, investment, and savings advice.
                Take control of your financial future with data-driven insights.
              </p>
            </div>
            <div className="social-links">
              <a
                href="https://www.facebook.com"
                target="_blank"
                rel="noopener noreferrer"
                className="social-icon"
              >
                <FontAwesomeIcon icon={faFacebookF} />
              </a>
              <a
                href="https://www.twitter.com"
                target="_blank"
                rel="noopener noreferrer"
                className="social-icon"
              >
                <FontAwesomeIcon icon={faTwitter} />
              </a>
              <a
                href="https://www.linkedin.com"
                target="_blank"
                rel="noopener noreferrer"
                className="social-icon"
              >
                <FontAwesomeIcon icon={faLinkedinIn} />
              </a>
              <a
                href="https://www.instagram.com"
                target="_blank"
                rel="noopener noreferrer"
                className="social-icon"
              >
                <FontAwesomeIcon icon={faInstagram} />
              </a>
            </div>
          </Col>

          <Col lg={2} md={6} sm={6} className="mb-4 mb-md-0">
            <h5 className="footer-heading">Company</h5>
            <ul className="footer-links">
              <li>
                <Link to="/about">About Us</Link>
              </li>
              <li>
                <Link to="/services">Services</Link>
              </li>
              <li>
                <Link to="/contact">Contact</Link>
              </li>
              <li>
                <a href="#">Careers</a>
              </li>
            </ul>
          </Col>

          <Col lg={2} md={6} sm={6} className="mb-4 mb-md-0">
            <h5 className="footer-heading">Resources</h5>
            <ul className="footer-links">
              <li>
                <a href="#">Blog</a>
              </li>
              <li>
                <a href="#">Help Center</a>
              </li>
              <li>
                <a href="#">Privacy Policy</a>
              </li>
              <li>
                <a href="#">Terms of Service</a>
              </li>
            </ul>
          </Col>

          <Col lg={4} md={6} className="mb-4 mb-md-0">
            <h5 className="footer-heading">Subscribe to Our Newsletter</h5>
            <p>
              Get the latest financial tips and updates delivered to your inbox.
            </p>
            <Form className="newsletter-form">
              <div className="d-flex">
                <Form.Control
                  type="email"
                  placeholder="Your email address"
                  className="newsletter-input"
                />
                <Button type="submit" className="newsletter-button">
                  Subscribe
                </Button>
              </div>
            </Form>
          </Col>
        </Row>

        <hr className="footer-divider" />

        <div className="footer-bottom">
          <p className="copyright">
            &copy; {currentYear} Financial AI Advisor. All rights reserved.
          </p>
          <div className="footer-bottom-links">
            <a href="#">Privacy Policy</a>
            <a href="#">Terms of Service</a>
            <a href="#">Cookies</a>
          </div>
        </div>
      </Container>
    </footer>
  );
}

export default Footer;
