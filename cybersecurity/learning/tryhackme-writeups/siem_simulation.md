# TryHackMe - SIEM Simulation

**Difficulty:** Beginner  
**Category:** Security Operations Center (SOC)  
**Date Completed:** 09/09/2025
**Platform:** TryHackMe  
**Skills:** SIEM Analysis, Threat Intelligence, Incident Response

## Overview

This room provides hands-on experience with SIEM (Security Information and Event Management) operations, simulating real-world SOC analyst workflows. The objective is to analyze security alerts, identify malicious activity, and perform threat intelligence investigations using open-source databases.

## Learning Objectives

- Understanding SIEM dashboard analysis
- Alert triage and investigation procedures
- Threat intelligence gathering techniques
- IP reputation analysis using OSINT tools
- Incident escalation processes

## SIEM Dashboard Analysis

### Initial Assessment

**Dashboard Overview:**
- **Operations Information:** 40% (134 events)
- **Security Attacks:** 30% (102 events)  
- **Security Suspicious:** 30% (102 events)

**Geographic Distribution of Events:**
| Country | Event Count | Risk Level |
|---------|-------------|------------|
| UK      |    60       | Low        |
| US      |    30       | Medium     |
| Brazil  |    21       | Medium     |    
| Russia  |    15       | High       |
| Korea   |    17       | Critical   |
| China   |    4        | High       |

### Alert Log Investigation

**Timeline Analysis (October 9th, 2025):**

```
01:26:08 - Unauthorized connection attempt detected from IP 143.110.250.149 to port 22
01:29:12 - Successful SSH authentication attempt to port 22 from IP 143.110.250.149
10:59:07 - Logon Failure: Account Password Expired (Event ID 535)
11:09:29 - Multiple failed login attempts from John Doe
11:09:42 - User John Doe logged in successfully (Event ID 4624)
```

## Threat Analysis

### Malicious IP Identification

**Primary IOC (Indicator of Compromise):**
- **IP Address:** `143.110.250.149`
- **Attack Pattern:** SSH brute-force → Successful authentication
- **Timeline:** 3-minute window between attempt and success

### Threat Intelligence Investigation

**IP Reputation Analysis:**
```
Target IP: 143.110.250.149
Database: ip-scanner.thm (simulated threat intelligence platform)

Results:
├── Malicious Confidence: 100%
├── Classification: Malicious
├── ISP: China Mobile Communications Corporation  
├── Domain: chinamobileltd.thm
├── Country: China
└── City: Zhenjiang, Jiangsu
```

**Additional OSINT Sources Referenced:**
- AbuseIPDB
- Cisco Talos Intelligence
- (Note: These would be used in real-world scenarios)

## Attack Vector Analysis

### SSH Compromise Sequence

1. **Initial Reconnaissance** (01:26:08)
   - Unauthorized connection attempt to port 22
   - Scanning/probing phase

2. **Successful Breach** (01:29:12)
   - SSH authentication successful
   - 3-minute window suggests automated attack tools
   - Possible credential stuffing or brute-force success

3. **Lateral Movement Potential**
   - Successful SSH access provides shell access
   - Risk of privilege escalation
   - Potential for persistent access establishment

## Risk Assessment

### Severity: **HIGH**

**Impact Analysis:**
- **Confidentiality:** Unauthorized access to system resources
- **Integrity:** Potential system modification capabilities
- **Availability:** Risk of service disruption or ransomware
- **Compliance:** Potential regulatory violations

**MITRE ATT&CK Framework Mapping:**
- **TA0001 - Initial Access:** Valid Accounts (T1078)
- **TA0002 - Execution:** Command and Scripting Interpreter (T1059)
- **TA0003 - Persistence:** SSH Authorized Keys (T1098.004)

## Incident Response Actions

### Immediate Response (Executed)
1. **Alert Investigation:**  Completed
2. **Threat Intelligence:**  IP reputation confirmed malicious
3. **IOC Documentation:**  IP address catalogued
4. **Escalation:**  Prepared for staff notification



## Tools and Techniques Used

**SIEM Analysis:**
- Dashboard monitoring and alert triage
- Log correlation and timeline analysis
- Event pattern recognition

**Threat Intelligence:**
- IP reputation databases
- Geolocation analysis
- ISP and infrastructure mapping

**Documentation:**
- IOC (Indicators of Compromise) cataloging
- Timeline reconstruction
- Risk assessment frameworks

## Key Learning Outcomes

### Technical Skills Developed
- SIEM dashboard navigation and interpretation
- Alert prioritization based on risk factors
- Threat intelligence gathering methodologies
- Incident timeline reconstruction

### SOC Analyst Competencies
- **Alert Triage:** Prioritizing high-risk events from noise
- **Investigation:** Following evidence trails systematically  
- **Documentation:** Maintaining clear incident records
- **Escalation:** Knowing when to involve senior analysts

### Cybersecurity Frameworks Applied
- **NIST Incident Response:** Preparation → Detection → Analysis → Containment
- **MITRE ATT&CK:** Tactic and technique identification
- **Cyber Kill Chain:** Understanding attack progression


## References

- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [MITRE ATT&CK Framework](https://attack.mitre.org/)
- [AbuseIPDB](https://www.abuseipdb.com/)
- [Cisco Talos Intelligence](https://talosintelligence.com/)
- [SANS SOC Survey](https://www.sans.org/white-papers/)

---

**Note:** This writeup documents a simulated exercise for educational purposes.
