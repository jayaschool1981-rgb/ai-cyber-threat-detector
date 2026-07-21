'use client';

import { useState, useEffect } from 'react';
import styles from './page.module.css';

import { getApiBaseUrl } from '@/lib/api';

const API_BASE_URL = getApiBaseUrl();

interface Alert {
  id: number;
  prediction: string;
  confidence: number | null;
  destination_port: number | null;
  flow_duration: number | null;
  total_fwd_packets: number | null;
  total_backward_packets: number | null;
  created_at: string;
  user_id: number | null;
}

export default function Dashboard() {
  // Auth state
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isRegisterMode, setIsRegisterMode] = useState(false);
  const [token, setToken] = useState('');
  const [username, setUsername] = useState('');
  
  // Inputs
  const [authUsername, setAuthUsername] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  
  // App views
  const [activeTab, setActiveTab] = useState<'dashboard' | 'simulator' | 'logs'>('dashboard');
  const [errorMsg, setErrorMsg] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  // Simulator state
  const [port, setPort] = useState('80');
  const [duration, setDuration] = useState('12000');
  const [fwdPackets, setFwdPackets] = useState('2');
  const [bwdPackets, setBwdPackets] = useState('3');
  const [simLoading, setSimLoading] = useState(false);
  const [simResult, setSimResult] = useState<{ prediction: string; confidence: number | null } | null>(null);

  // Data logs state
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [stats, setStats] = useState({
    total: 0,
    benign: 0,
    malicious: 0,
    ratio: '0%',
    avgConfidence: '0.00'
  });

  // Restore auth from storage
  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    const savedUser = localStorage.getItem('username');
    if (savedToken && savedUser) {
      setToken(savedToken);
      setUsername(savedUser);
      setIsLoggedIn(true);
    }
  }, []);

  // Fetch alerts and calculate statistics
  const fetchAlertData = async (authToken: string) => {
    if (!authToken) return;
    try {
      const res = await fetch(`${API_BASE_URL}/alerts?limit=50`, {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      });
      if (res.status === 200) {
        const data: Alert[] = await res.json();
        setAlerts(data);
        
        // Calculate statistics
        if (data.length > 0) {
          const total = data.length;
          const benign = data.filter(a => a.prediction.toUpperCase() === 'BENIGN').length;
          const malicious = total - benign;
          const ratio = total > 0 ? `${Math.round((malicious / total) * 100)}%` : '0%';
          
          let confSum = 0;
          let confCount = 0;
          data.forEach(a => {
            if (a.confidence !== null) {
              confSum += a.confidence;
              confCount++;
            }
          });
          const avgConfidence = confCount > 0 ? (confSum / confCount).toFixed(4) : '0.0000';

          setStats({
            total,
            benign,
            malicious,
            ratio,
            avgConfidence
          });
        } else {
          setStats({ total: 0, benign: 0, malicious: 0, ratio: '0%', avgConfidence: '0.0000' });
        }
      } else if (res.status === 401) {
        handleLogout();
      }
    } catch (err) {
      console.error('Failed to fetch dashboard alerts:', err);
    }
  };

  // Poll alerts data when logged in
  useEffect(() => {
    if (isLoggedIn && token) {
      fetchAlertData(token);
      const interval = setInterval(() => fetchAlertData(token), 5000);
      return () => clearInterval(interval);
    }
  }, [isLoggedIn, token]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg('');
    try {
      const res = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: authUsername, password: authPassword })
      });
      const data = await res.json();
      if (res.status === 200) {
        localStorage.setItem('token', data.access_token);
        localStorage.setItem('username', authUsername);
        setToken(data.access_token);
        setUsername(authUsername);
        setIsLoggedIn(true);
        setAuthPassword('');
        setAuthUsername('');
      } else {
        setErrorMsg(data.detail || 'Login failed. Invalid credentials.');
      }
    } catch (err) {
      setErrorMsg('Unable to connect to the backend server.');
    }
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg('');
    setSuccessMsg('');
    try {
      const res = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: authUsername, password: authPassword })
      });
      const data = await res.json();
      if (res.status === 201 || res.status === 200) {
        setSuccessMsg('Registration successful! You can now log in.');
        setIsRegisterMode(false);
        setAuthPassword('');
      } else {
        setErrorMsg(data.detail || 'Registration failed.');
      }
    } catch (err) {
      setErrorMsg('Unable to connect to the backend server.');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    setToken('');
    setUsername('');
    setIsLoggedIn(false);
    setAlerts([]);
    setSimResult(null);
  };

  // Phase 4 Toast & Copy state
  const [toastMsg, setToastMsg] = useState('');
  const [copied, setCopied] = useState(false);
  const [validationError, setValidationError] = useState('');

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setToastMsg('Copied to clipboard!');
    setTimeout(() => {
      setCopied(false);
      setToastMsg('');
    }, 2500);
  };

  const runPrediction = async (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError('');
    setErrorMsg('');

    // Pre-submit Input Validation
    const portNum = Number(port);
    if (isNaN(portNum) || portNum < 1 || portNum > 65535) {
      setValidationError('Destination Port must be a valid integer between 1 and 65535.');
      return;
    }

    const durationNum = Number(duration);
    if (isNaN(durationNum) || durationNum < 0) {
      setValidationError('Flow Duration must be a non-negative number.');
      return;
    }

    setSimLoading(true);
    setSimResult(null);

    const record = {
      'Destination Port': portNum,
      'Flow Duration': durationNum,
      'Total Fwd Packets': Number(fwdPackets) || 0,
      'Total Backward Packets': Number(bwdPackets) || 0
    };

    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { 'Authorization': `Bearer ${token}` } : {})
        },
        body: JSON.stringify(record)
      });
      const data = await res.json();
      if (res.status === 200) {
        const resultObj = Array.isArray(data) ? data[0] : (data.data || data);
        setSimResult(resultObj);
        if (token) fetchAlertData(token);
      } else {
        setErrorMsg(data.detail || 'Inference call failed.');
      }
    } catch (err) {
      // Fallback prediction call to Express backend endpoint
      try {
        const fallbackRes = await fetch('http://localhost:5000/api/v1/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            destinationPort: portNum,
            flowDuration: durationNum,
            totalFwdPackets: Number(fwdPackets) || 0,
            totalBwdPackets: Number(bwdPackets) || 0
          })
        });
        const fallbackData = await fallbackRes.json();
        if (fallbackRes.ok && fallbackData.data) {
          setSimResult(fallbackData.data);
        } else {
          setErrorMsg('Prediction service returned an error.');
        }
      } catch {
        setErrorMsg('Prediction server error. Ensure local backend is running.');
      }
    } finally {
      setSimLoading(false);
    }
  };

  // Screen: Logged Out (Auth forms)
  if (!isLoggedIn) {
    return (
      <div className={styles.container}>
        <div className={styles.authWrapper}>
          <div className={styles.authCard}>
            <div className={styles.logoArea} style={{ justifyContent: 'center', marginBottom: '1.5rem' }}>
              <div className={styles.logoDot} />
              <div className={styles.logoText}>SECURITY ENGINE</div>
            </div>
            
            <h2 className={styles.authTitle}>
              {isRegisterMode ? 'Create Account' : 'Gateway Login'}
            </h2>
            <p className={styles.authSubtitle}>
              {isRegisterMode ? 'Register to view cybersecurity logs and run model predictions' : 'Enter credentials to access threat detection engine'}
            </p>

            <div style={{ 
              display: 'flex', 
              alignItems: 'flex-start', 
              gap: '0.625rem', 
              background: 'rgba(59, 130, 246, 0.04)', 
              border: '1px solid rgba(59, 130, 246, 0.12)', 
              borderRadius: '0.5rem', 
              padding: '0.75rem 1rem', 
              marginBottom: '1.5rem',
              fontSize: '0.75rem',
              color: '#9ca3af',
              lineHeight: '1.4',
              textAlign: 'left'
            }}>
              <span style={{ fontSize: '1rem', marginTop: '-0.1rem' }}>🔒</span>
              <span>
                <strong>Secure Sandbox:</strong> This is an isolated testing system. All registered credentials and flow simulation logs are processed locally. Your data is completely safe and no production traffic is monitored.
              </span>
            </div>

            {errorMsg && <div className={styles.errorText}>{errorMsg}</div>}
            {successMsg && <div className={styles.successText}>{successMsg}</div>}

            <form onSubmit={isRegisterMode ? handleRegister : handleLogin}>
              <div className={styles.authFormGroup}>
                <div>
                  <label className={styles.formLabel}>Username</label>
                  <input
                    type="text"
                    required
                    value={authUsername}
                    onChange={(e) => setAuthUsername(e.target.value)}
                    className={styles.formInput}
                    placeholder="e.g. administrator"
                    style={{ width: '100%' }}
                  />
                </div>
                <div>
                  <label className={styles.formLabel}>Password</label>
                  <input
                    type="password"
                    required
                    value={authPassword}
                    onChange={(e) => setAuthPassword(e.target.value)}
                    className={styles.formInput}
                    placeholder="••••••••"
                    style={{ width: '100%' }}
                  />
                </div>
              </div>

              <button type="submit" className={styles.primaryButton}>
                {isRegisterMode ? 'Sign Up' : 'Authenticate'}
              </button>
            </form>

            <button
              onClick={() => {
                setIsRegisterMode(!isRegisterMode);
                setErrorMsg('');
                setSuccessMsg('');
              }}
              className={styles.secondaryButton}
            >
              {isRegisterMode ? 'Already have an account? Login' : 'Register New User'}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Screen: Logged In (Dashboard Panels)
  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.logoArea}>
          <div className={styles.logoDot} />
          <div className={styles.logoText}>THREAT DETECTOR</div>
        </div>
        
        <nav className={styles.navGroup}>
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`${styles.navButton} ${activeTab === 'dashboard' ? styles.activeNav : ''}`}
          >
            Dashboard
          </button>
          <button
            onClick={() => setActiveTab('simulator')}
            className={`${styles.navButton} ${activeTab === 'simulator' ? styles.activeNav : ''}`}
          >
            Simulator
          </button>
          <button
            onClick={() => setActiveTab('logs')}
            className={`${styles.navButton} ${activeTab === 'logs' ? styles.activeNav : ''}`}
          >
            Alert Logs
          </button>
        </nav>

        <div className={styles.userInfo}>
          <span className={styles.usernameTag}>👤 {username}</span>
          <button onClick={handleLogout} className={styles.logoutBtn}>Logout</button>
        </div>
      </header>

      <main className={styles.mainContent}>
        
        {/* Metrics Overview */}
        <section className={styles.statsRow}>
          <div className={styles.statCard}>
            <span className={styles.statLabel}>System Status</span>
            <span className={`${styles.statValue} ${stats.malicious > 0 ? styles.statValueAlert : styles.statValueHealthy}`}>
              {stats.malicious > 0 ? '⚠️ AT RISK' : '🛡️ SECURE'}
            </span>
            <span className={styles.statDesc}>Engine monitoring active</span>
          </div>
          <div className={styles.statCard}>
            <span className={styles.statLabel}>Total Scanned Flows</span>
            <span className={styles.statValue}>{stats.total}</span>
            <span className={styles.statDesc}>Recorded alerts in database</span>
          </div>
          <div className={styles.statCard}>
            <span className={styles.statLabel}>Threat Ratio</span>
            <span className={`${styles.statValue} ${stats.malicious > 0 ? styles.statValueAlert : ''}`}>
              {stats.ratio}
            </span>
            <span className={styles.statDesc}>{stats.malicious} malicious detections</span>
          </div>
          <div className={styles.statCard}>
            <span className={styles.statLabel}>Avg Model Confidence</span>
            <span className={styles.statValue}>{stats.avgConfidence}</span>
            <span className={styles.statDesc}>ONNX performance coefficient</span>
          </div>
        </section>

        {errorMsg && <div className={styles.errorText} style={{ marginBottom: '2rem' }}>{errorMsg}</div>}

        {/* Tab: Dashboard Visual Summary */}
        {activeTab === 'dashboard' && (
          <div className={styles.panelCard}>
            <h3 className={styles.panelTitle}>📊 Enterprise Threat Vector Summary</h3>
            <p style={{ color: '#9ca3af', marginBottom: '2rem', fontSize: '0.9rem' }}>
              Real-time cyber threat analytics powered by **ONNX Runtime (Pipeline v1.0.0)**.
            </p>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '2rem' }}>
              <div style={{ background: 'rgba(3, 7, 18, 0.4)', padding: '1.5rem', borderRadius: '0.5rem', border: '1px solid rgba(255,255,255,0.04)' }}>
                <h4 style={{ fontWeight: '600', marginBottom: '1rem', color: '#e5e7eb' }}>Scan Classification</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
                      <span>Benign flows</span>
                      <span style={{ color: '#34d399' }}>{stats.benign}</span>
                    </div>
                    <div style={{ height: '0.5rem', background: '#1f2937', borderRadius: '9999px', overflow: 'hidden' }}>
                      <div style={{ height: '100%', background: '#10b981', width: stats.total > 0 ? `${(stats.benign / stats.total) * 100}%` : '0%' }} />
                    </div>
                  </div>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
                      <span>Threats/Attacks</span>
                      <span style={{ color: '#f87171' }}>{stats.malicious}</span>
                    </div>
                    <div style={{ height: '0.5rem', background: '#1f2937', borderRadius: '9999px', overflow: 'hidden' }}>
                      <div style={{ height: '100%', background: '#ef4444', width: stats.total > 0 ? `${(stats.malicious / stats.total) * 100}%` : '0%' }} />
                    </div>
                  </div>
                </div>
              </div>

              <div style={{ background: 'rgba(3, 7, 18, 0.4)', padding: '1.5rem', borderRadius: '0.5rem', border: '1px solid rgba(255,255,255,0.04)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <h4 style={{ fontWeight: '600', marginBottom: '0.5rem', color: '#e5e7eb' }}>Active Infrastructure Details</h4>
                <p style={{ fontSize: '0.875rem', color: '#9ca3af', lineHeight: '1.6' }}>
                  Model execution runtime is highly optimized using **ONNX (Open Neural Network Exchange)**. Standard RFC protocols are monitored on ingress ports (Port 80/443, etc.). Detections log user context details, confidence metrics, and connection characteristics to PostgreSQL.
                </p>
                <div style={{ display: 'flex', gap: '2rem', marginTop: '1.25rem' }}>
                  <div>
                    <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>Database Engine</div>
                    <div style={{ fontSize: '0.875rem', fontWeight: '600', color: '#3b82f6' }}>PostgreSQL</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>Caching & Limit State</div>
                    <div style={{ fontSize: '0.875rem', fontWeight: '600', color: '#3b82f6' }}>Redis Active</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>Inference Device</div>
                    <div style={{ fontSize: '0.875rem', fontWeight: '600', color: '#3b82f6' }}>CPU (Optimized)</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab: Real-time Simulator Form */}
        {activeTab === 'simulator' && (
          <div className={styles.panelCard}>
            <h3 className={styles.panelTitle}>🚀 Network Flow Simulator</h3>
            <p style={{ color: '#9ca3af', marginBottom: '2rem', fontSize: '0.9rem' }}>
              Input connection flow parameters to execute real-time threat detection.
            </p>
            
            {validationError && (
              <div className={styles.alertBanner}>
                <span>⚠️ {validationError}</span>
              </div>
            )}

            {toastMsg && (
              <div className={styles.toast}>
                ✓ {toastMsg}
              </div>
            )}

            <div className={styles.simulatorGrid}>
              <form onSubmit={runPrediction} className={styles.formGrid}>
                <div>
                  <label className={styles.formLabel}>Destination Port (1 - 65535)</label>
                  <input
                    type="number"
                    required
                    min="1"
                    max="65535"
                    value={port}
                    onChange={(e) => setPort(e.target.value)}
                    className={styles.formInput}
                    placeholder="e.g. 80"
                    style={{ width: '100%' }}
                  />
                  <div className={styles.charCounter}>{port.length} chars</div>
                </div>
                <div>
                  <label className={styles.formLabel}>Flow Duration (ms)</label>
                  <input
                    type="number"
                    required
                    min="0"
                    value={duration}
                    onChange={(e) => setDuration(e.target.value)}
                    className={styles.formInput}
                    placeholder="e.g. 12000"
                    style={{ width: '100%' }}
                  />
                  <div className={styles.charCounter}>{duration.length} chars</div>
                </div>
                <div>
                  <label className={styles.formLabel}>Total Forward Packets</label>
                  <input
                    type="number"
                    required
                    min="0"
                    value={fwdPackets}
                    onChange={(e) => setFwdPackets(e.target.value)}
                    className={styles.formInput}
                    placeholder="e.g. 2"
                    style={{ width: '100%' }}
                  />
                  <div className={styles.charCounter}>{fwdPackets.length} chars</div>
                </div>
                <div>
                  <label className={styles.formLabel}>Total Backward Packets</label>
                  <input
                    type="number"
                    required
                    min="0"
                    value={bwdPackets}
                    onChange={(e) => setBwdPackets(e.target.value)}
                    className={styles.formInput}
                    placeholder="e.g. 3"
                    style={{ width: '100%' }}
                  />
                  <div className={styles.charCounter}>{bwdPackets.length} chars</div>
                </div>
                
                <div style={{ gridColumn: 'span 2', marginTop: '1rem' }}>
                  <button type="submit" disabled={simLoading} className={styles.primaryButton}>
                    {simLoading ? 'Executing Classifier...' : 'Run Detection'}
                  </button>
                </div>
              </form>

              <div className={styles.simResultBox}>
                {simResult ? (
                  <div className={styles.saasCard}>
                    <div className={styles.saasCardHeader}>
                      <span className={styles.resultTitle}>Enterprise Threat Intelligence</span>
                      <button 
                        onClick={() => copyToClipboard(JSON.stringify(simResult, null, 2))}
                        className={styles.copyBtn}
                      >
                        {copied ? '✓ Copied!' : '📋 Copy JSON'}
                      </button>
                    </div>

                    <div className={`${styles.resultBadge} ${String(simResult.prediction || '').toUpperCase() === 'BENIGN' ? styles.badgeBenign : styles.badgeMalicious}`}>
                      CLASSIFICATION: {simResult.prediction}
                    </div>

                    <div style={{ fontSize: '0.85rem', color: '#d1d5db', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                      {simResult.confidence !== undefined && (
                        <div><strong>Confidence:</strong> {typeof simResult.confidence === 'number' && simResult.confidence <= 1 ? (simResult.confidence * 100).toFixed(2) : simResult.confidence}%</div>
                      )}
                      {(simResult as any).riskLevel && (
                        <div><strong>Risk Level:</strong> <span style={{ color: (simResult as any).riskLevel === 'CRITICAL' || (simResult as any).riskLevel === 'HIGH' ? '#f87171' : '#34d399' }}>{(simResult as any).riskLevel}</span></div>
                      )}
                      {(simResult as any).threatVector && (
                        <div><strong>Threat Vector:</strong> {(simResult as any).threatVector}</div>
                      )}
                      {(simResult as any).recommendedAction && (
                        <div><strong>Recommended Action:</strong> {(simResult as any).recommendedAction}</div>
                      )}
                      {(simResult as any).providerUsed && (
                        <div><strong>Engine Tier:</strong> <span style={{ color: '#60a5fa' }}>{(simResult as any).providerUsed}</span></div>
                      )}
                    </div>

                    <div className={styles.charCounter}>
                      Payload: {JSON.stringify(simResult).length} characters
                    </div>
                  </div>
                ) : (
                  <span className={styles.noResultText}>
                    {simLoading ? 'Querying AI provider circuit...' : 'Enter flow logs and trigger simulator scan.'}
                  </span>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Tab: Historical Logs Table */}
        {activeTab === 'logs' && (
          <div className={styles.panelCard}>
            <h3 className={styles.panelTitle}>📋 Security Event Logs</h3>
            <p style={{ color: '#9ca3af', marginBottom: '2rem', fontSize: '0.9rem' }}>
              Historical alerts registered in the database for the active threat agent.
            </p>

            <div className={styles.tableContainer}>
              <table className={styles.logTable}>
                <thead>
                  <tr>
                    <th>Alert ID</th>
                    <th>Classification</th>
                    <th>Confidence</th>
                    <th>Dest Port</th>
                    <th>Duration</th>
                    <th>Fwd Pkts</th>
                    <th>Bwd Pkts</th>
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {alerts.length > 0 ? (
                    alerts.map((alert) => (
                      <tr key={alert.id}>
                        <td style={{ fontFamily: 'monospace', color: '#6b7280' }}>#{alert.id}</td>
                        <td>
                          <span className={`${styles.logBadge} ${alert.prediction.toUpperCase() === 'BENIGN' ? styles.logBadgeBenign : styles.logBadgeMalicious}`}>
                            {alert.prediction}
                          </span>
                        </td>
                        <td>{alert.confidence !== null ? `${(alert.confidence * 100).toFixed(2)}%` : 'N/A'}</td>
                        <td>{alert.destination_port ?? 'N/A'}</td>
                        <td>{alert.flow_duration ?? 'N/A'}</td>
                        <td>{alert.total_fwd_packets ?? 'N/A'}</td>
                        <td>{alert.total_backward_packets ?? 'N/A'}</td>
                        <td>{new Date(alert.created_at).toLocaleString()}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={8} style={{ textAlign: 'center', color: '#6b7280', padding: '2rem 0' }}>
                        No records logged in the database. Run predictions in the simulator to generate logs.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}
