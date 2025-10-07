# Home Office Network Asset Inventory

## Project Overview
This project demonstrates **Asset Management** principles for a small business home office network. The inventory identifies and classifies network-connected devices based on their security sensitivity levels.

---

## Scenario
Operating a small business from home requires careful management of network-connected devices. This inventory helps identify which devices contain sensitive information requiring extra protection.

---

## Network Device Inventory

| # | Asset | Network Access | Owner | Location | Notes | Sensitivity |
|---|-------|----------------|-------|----------|-------|-------------|
| 1 | **Network Router** | Continuous | Internet Service Provider (ISP) | On-premises | Has 2.4 GHz and 5 GHz connections. All devices connect to 5 GHz frequency. | 🔴 **Confidential** |
| 2 | **Desktop Computer** | Occasional | Homeowner | On-premises | Contains private information, like photos and business documents. | 🔴 **Restricted** |
| 3 | **Guest Smartphone** | Occasional | Friend | On/Off-premises | Connects to home network occasionally. | 🟡 **Internal-only** |
| 4 | **External Hard Drive** | Occasional | Homeowner | On-premises (home office) | Contains business backups and sensitive financial records. Network-accessible for backups. | 🔴 **Restricted** |
| 5 | **Wireless Printer** | Continuous | Homeowner | On-premises (home office) | Network-enabled printer accessible by all home devices. Stores print jobs temporarily. | 🔴 **Confidential** |
| 6 | **Smart TV/Streaming Device** | Continuous | Homeowner | On-premises (living room) | Connected to 5GHz network. Has access to streaming accounts and browsing history. | 🟡 **Internal-only** |

---

## Sensitivity Classification System

| Category | Access Level | Description | Risk Level |
|----------|-------------|-------------|------------|
| 🔴 **Restricted** | Need-to-know | Highly sensitive business data | Critical |
| 🔴 **Confidential** | Limited users | Sensitive personal/business info | High |
| 🟡 **Internal-only** | On-premises users | Internal household access only | Medium |
| 🟢 **Public** | Anyone | Publicly accessible information | Low |

---

## Device Analysis

### Critical Assets (Require Maximum Protection)
- **External Hard Drive**: Contains financial records and business-critical backups
- **Desktop Computer**: Stores sensitive business and personal documents

### High-Priority Assets
- **Network Router**: Gateway to all network resources
- **Wireless Printer**: Can cache sensitive documents

### Medium-Priority Assets
- **Smart TV**: Contains personal accounts but not business data
- **Guest Smartphone**: Temporary access, limited permissions needed

---

## Security Recommendations

Based on this inventory, the following security measures are recommended:

1. **Network Segmentation**
   - Create guest network for visitor devices
   - Isolate smart home devices on separate VLAN

2. **Access Control**
   - Enable WPA3 encryption on router
   - Implement strong password policy
   - Regular password updates for critical devices

3. **Data Protection**
   - Encrypt external hard drive
   - Enable firewall on desktop computer
   - Regular backups of business-critical data

4. **Monitoring**
   - Regular review of connected devices
   - Monitor unusual network activity
   - Update firmware regularly

---

## Risk Assessment Matrix

| Device | Confidentiality Impact | Integrity Impact | Availability Impact | Overall Risk |
|--------|------------------------|------------------|--------------------|--------------| 
| External Hard Drive | High | High | High | **Critical** |
| Desktop | High | Medium | High | **High** |
| Router | Medium | High | High | **High** |
| Printer | Medium | Low | Medium | **Medium** |
| Smart TV | Low | Low | Low | **Low** |
| Guest Phone | Low | Low | Low | **Low** |

---

## Key Takeaways

- **6 devices** identified and classified
- **2 critical assets** requiring maximum protection
- **2 high-priority assets** needing strong security controls
- **2 medium/low priority assets** with standard security measures

---

## References

- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [ISO 27001 Asset Management](https://www.iso.org/isoiec-27001-information-security.html)
- Home Network Security Best Practices

---


### 📄 License

This project is part of a cybersecurity course assignment and is for educational purposes.