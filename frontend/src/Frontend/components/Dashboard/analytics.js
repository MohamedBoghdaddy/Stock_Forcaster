import { useEffect, useState } from "react";
import { Bar } from "react-chartjs-2";
import { Spinner, Table } from "react-bootstrap";
import { BsFileEarmarkText, BsDownload } from "react-icons/bs";
import axios from "axios";
import useDashboard from "../../../hooks/useDashboard";
import "bootstrap/dist/css/bootstrap.min.css";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import "../styles/Analytics.css";
import { BsArrowUp, BsArrowDown } from "react-icons/bs";
import Badge from "../../../ui/badge/Badge";
import amazon from "../../../assets/icons8-amazon-100.svg";
import apple from "../../../assets/icons8-apple-inc-100.svg";
import google from "../../../assets/icons8-google-96.svg";
import meta from "../../../assets/icons8-meta-96.svg";
import tesla from "../../../assets/icons8-tesla-96.svg";
import netflix from "../../../assets/icons8-netflix-80.svg";

const stockSymbols = [
  "AAPL",
  "META",
  "AMZN",
  "NFLX",
  "GOOGL",
  "MSFT",
  "TSLA",
  "NVDA",
  "BRK-B",
  "JPM",
  "V",
  "JNJ",
  "WMT",
  "UNH",
  "PG",
];

const stockIcons = {
  AAPL: apple,
  AMZN: amazon,
  GOOGL: google,
  META: meta,
  TSLA: tesla,
  NFLX: netflix,
};

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const API_URL =
  process.env.REACT_APP_API_URL || "http://localhost:4000/api/analytics";

export default function AnalyticsReport() {
  const { state = {}, fetchProfile, handleDownload } = useDashboard() || {};
  const { profile = {}, reports = [] } = state;
  const [loadingAnalytics, setLoadingAnalytics] = useState(true);
  const [riskToleranceData, setRiskToleranceData] = useState([]);
  const [lifestyleData, setLifestyleData] = useState([]);
  const [metrics, setMetrics] = useState({});

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const [riskRes, lifestyleRes] = await Promise.all([
          axios.get(`${API_URL}/risk-tolerance`),
          axios.get(`${API_URL}/lifestyle`),
        ]);
        setRiskToleranceData(Array.isArray(riskRes.data) ? riskRes.data : []);
        setLifestyleData(
          Array.isArray(lifestyleRes.data) ? lifestyleRes.data : []
        );
      } catch (error) {
        console.error("❌ Error fetching analytics:", error);
      } finally {
        setLoadingAnalytics(false);
      }
    };

    if (fetchProfile) fetchProfile();
    fetchAnalytics();
  }, [fetchProfile]);

  useEffect(() => {
    const fetchStockData = async () => {
      const results = {};

      for (const symbol of stockSymbols) {
        try {
          const res = await axios.get("http://localhost:8000/historical", {
            params: { symbol, period: "1d" },
          });
          const data = res.data.data;
          const closePrices = data
            .map((d) => d.Close ?? d.close)
            .filter((p) => typeof p === "number");

          if (closePrices.length >= 2) {
            const latest = closePrices[closePrices.length - 1];
            const previous = closePrices[closePrices.length - 2];
            const change = ((latest - previous) / previous) * 100;
            results[symbol] = {
              latest: Number(latest.toFixed(2)),
              change: Number(change.toFixed(2)),
            };
          }
        } catch (err) {
          console.error(`❌ Failed to fetch data for ${symbol}:`, err);
        }
      }

      setMetrics(results);
    };

    fetchStockData();
  }, []);

  const analyticsChartData = {
    labels: riskToleranceData.map((item) => item._id || "Unknown"),
    datasets: [
      {
        label: "Users per Risk Tolerance",
        data: riskToleranceData.map((item) => item.totalUsers || 0),
        backgroundColor: ["#f1c40f", "#27ae60", "#2980b9"],
      },
    ],
  };

  const chartOptions = {
    scales: {
      y: { beginAtZero: true, ticks: { stepSize: 1 } },
    },
  };

  return (
    <div className="analytic-container">
      <div className="main">
        <div className="main-top">
          <h1>Financial & Investment Analytics</h1>
        </div>

        <h2>Risk Tolerance Distribution</h2>
        <div
          className="chart-container"
          style={{ width: "60%", margin: "auto" }}
        >
          {loadingAnalytics ? (
            <Spinner animation="border" role="status">
              <span className="sr-only">Loading...</span>
            </Spinner>
          ) : (
            <Bar data={analyticsChartData} options={chartOptions} />
          )}
        </div>

        {profile.name ? (
          <>
            <h2>User Profile</h2>
            <Table striped bordered hover>
              <tbody>
                <tr>
                  <td>Name</td>
                  <td>{profile.name}</td>
                </tr>
                <tr>
                  <td>Email</td>
                  <td>{profile.email}</td>
                </tr>
              </tbody>
            </Table>
          </>
        ) : (
          <p>User profile not available.</p>
        )}

        <h2>Lifestyle Preferences</h2>
        {lifestyleData.length > 0 ? (
          <Table striped bordered hover>
            <thead>
              <tr>
                <th>Lifestyle Type</th>
                <th>Users</th>
              </tr>
            </thead>
            <tbody>
              {lifestyleData.map((item) => (
                <tr key={item._id}>
                  <td>{item._id || "Unknown"}</td>
                  <td>{item.totalUsers || 0}</td>
                </tr>
              ))}
            </tbody>
          </Table>
        ) : (
          <p>No lifestyle data available.</p>
        )}

        <h2>Tracked Stocks</h2>
        <div className="grid-container">
          {Object.entries(metrics).map(([symbol, { latest, change }]) => (
            <div key={symbol} className="metric-card">
              <div className="metric-header">
                <div className="metric-info">
                  {stockIcons[symbol] && (
                    <img
                      src={stockIcons[symbol]}
                      alt={`${symbol} logo`}
                      className="symbol-icon"
                    />
                  )}
                  <h4 className="symbol-name">{symbol}</h4>
                </div>
                <Badge
                  className={change >= 0 ? "badge-success" : "badge-error"}
                >
                  {change >= 0 ? <BsArrowUp /> : <BsArrowDown />}
                  {Math.abs(change)}%
                </Badge>
              </div>
              <div className="price">${latest}</div>
            </div>
          ))}
        </div>

        <h2>Generated Reports</h2>
        <div className="report-container">
          <div className="report-list">
            {Array.isArray(reports) && reports.length > 0 ? (
              reports.map((report) => (
                <div key={report.id} className="report-card">
                  <BsFileEarmarkText className="report-icon" />
                  <div className="report-info">
                    <h4>{report.title}</h4>
                    <p>{new Date(report.date).toLocaleDateString()}</p>
                  </div>
                  <button
                    onClick={() => handleDownload(report.id)}
                    className="download-btn"
                  >
                    <BsDownload /> Download
                  </button>
                </div>
              ))
            ) : (
              <p>No reports available.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
