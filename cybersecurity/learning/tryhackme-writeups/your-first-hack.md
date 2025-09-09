# TryHackMe - Your First Hack

**Difficulty:** Beginner  
**Category:** Web Application Security  
**Date Completed:** 09/09/2025 
**Platform:** TryHackMe  

## Overview

"Your First Hack" is an introductory room that teaches fundamental web enumeration and exploitation concepts. The objective is to discover hidden pages on a fictional bank's website and exploit an access vulnerability to transfer money.

## Target Information

- **Target URL:** http://fakebank.thm
- **Scope:** Web application penetration testing
- **Goal:** Transfer $2000 from account 2276 to account 8881

## Methodology

### Phase 1: Reconnaissance & Enumeration

#### Directory Brute-forcing with Gobuster

**Command used:**
```bash
gobuster -u http://fakebank.thm -w wordlist.txt dir
```

**Parameters:**
- `-u`: Specifies target URL
- `-w`: Wordlist for brute-force attack
- `dir`: Directory enumeration mode

**Output:**
```
=====================================================
Gobuster v2.0.1              OJ Reeves (@TheColonial)
=====================================================
[+] Mode         : dir
[+] Url/Domain   : http://fakebank.thm/
[+] Threads      : 10
[+] Wordlist     : wordlist.txt
[+] Status codes : 200,204,301,302,307,403
[+] Timeout      : 10s
=====================================================
2024/05/21 10:04:38 Starting gobuster
=====================================================
/images (Status: 301)
/bank-transfer (Status: 200)
=====================================================
2024/05/21 10:04:44 Finished
=====================================================
```

**Enumeration Results:**
- `/images` - Images directory (Status: 301 - Redirect)
- `/bank-transfer` - **Critical page found!** (Status: 200 - Accessible)

### Phase 2: Exploitation

#### Vulnerability Identified
**IDOR (Insecure Direct Object Reference)**
- `/bank-transfer` page accessible without authentication
- No authorization controls on transfers
- Ability to transfer money from arbitrary accounts

#### Exploitation Steps
1. **Access to hidden page:**
   - Navigate to: `http://fakebank.thm/bank-transfer`
   
2. **Execute unauthorized transfer:**
   - Source account: `2276`
   - Destination account: `8881` (attacker's account)
   - Amount: `$2000`

3. **Success verification:**
   - Check account 8881 balance
   - Refresh page to see updates

## Impact Assessment

### Severity: **HIGH**

**Risks Identified:**
- **Financial losses:** Unauthorized transfers
- **Privacy breach:** Access to sensitive banking data
- **Compliance:** Violation of banking regulations (PCI DSS, GDPR)
- **Reputation:** Loss of customer trust

## Remediation

### Immediate Actions
1. **Remove public access** to `/bank-transfer` page
2. **Implement robust authentication** mechanisms
3. **Complete audit** of recent transactions

### Long-term Security Measures
1. **Access Control:**
   - Implement RBAC (Role-Based Access Control)
   - Multi-factor authentication for critical operations
   
2. **Input Validation:**
   - Server-side parameter validation
   - Authorization checks for every transaction
   
3. **Monitoring:**
   - Detailed transaction logging
   - Alerts for suspicious activities
   
4. **Security Testing:**
   - Regular penetration testing
   - Security-focused code reviews

## Tools Used

- **Gobuster v2.0.1** - Directory/file brute-forcing
- **Web Browser** - Manual testing and exploitation
- **Wordlist** - Directory enumeration

## Lessons Learned

### Technical Skills
- Basic use of Gobuster for directory enumeration
- Identification of exposed administrative pages
- Exploitation of IDOR vulnerabilities

### Security Concepts
- **Security by Obscurity** is not a valid defense
- Importance of proper access controls
- Impact of vulnerabilities in financial applications

### OWASP Top 10 Mapping
- **A01:2021 - Broken Access Control**
- **A05:2021 - Security Misconfiguration**


## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Gobuster GitHub](https://github.com/OJ/gobuster)
- [IDOR Explanation](https://portswigger.net/web-security/access-control/idor)

---

**Note:** This writeup is for educational purposes only. Always test on authorized systems.
