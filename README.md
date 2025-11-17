# Voice over New Radio (VoNR) Testbed on Private 5G

## Overview
This project develops a testbed for **Voice over New Radio (VoNR)** on a **private 5G network** using **Software Defined Radio (SDR)** and **open-source software**.  
Unlike previous studies that mainly focused on **network-centric metrics**, this work emphasizes the **Quality of Experience (QoE)** for end users.

## Objectives
- Build a functional testbed for VoNR over a private 5G network.  
- Identify the network conditions that deliver the best QoE.  
- Shift focus from traditional network performance to user experience.  

## Methodology

- The 5G core network was built using the Open5GS project from the official GitHub repository open5gs (config files)

- The radio access network was built using srsRAN from its GitHub repository, srsRAN (config files)

- SIM card programming was performed using a Rocketek smart card reader connected via USB with aid of pySIM tool from srsRAN from its GitHub repository

- Kamailio, IMS and pyHSS were deployed with aid of guidelines available on Kamailio and pyHSS official GitHub repository 
- Vary key network parameters: **latency**, **packet loss**, and **jitter**.
- Evaluate QoE using the **Mean Opinion Score (MOS)** under different conditions.
- Analyze results to determine thresholds and conditions that maximize QoE for VoNR.


## Expected Outcomes
- Insights into how latency, jitter, and packet loss affect user-perceived voice quality.  
- Recommendations for suitable network metrics and thresholds to achieve optimal QoE.  
- A reusable testbed framework for future 5G/VoNR experiments.  

## Technologies
- Software Defined Radio (SDR)  
- Open-source 5G software stack (srsRAN, Open5GS)  
- Linux-based test environment  



# **VoNR Testbed with Alternative VoIP Approach**

## **Project Objective**

The initial goal was to develop a **Voice over New Radio (VoNR) testbed** that supports IMS-based voice calls over a 5G network using srsRAN and open-source IMS components.

***

## **Challenges**

*   IMS registration could not be fully achieved despite deploying:
    *   **Kamailio** (PCSCF, ICSCF, SCSCF)
    *   **PyHSS** for HSS functionality
*   IMS bearer was unstable (connected/disconnected intermittently).
*   Time constraints prevented complete troubleshooting.

***

## **Alternative Approach Implemented**

Due to IMS limitations, an alternative **VoIP-over-5G solution** was deployed:

### **Setup**

*   **Two phones with Zoiper SIP clients**:
    *   Phone A: Connected via SDR + srsRAN (5G bearer).
    *   Phone B: Connected via Wi-Fi to the core.
*   **VoIP call initiated from Phone A to Phone B**.
*   ** Audio played into the Mic of phone A continuously**

### **Traffic Handling**

*   RTP traffic intercepted from **GTP-U tunnel**.
*   **Impairments applied using NetEm** (packet loss, delay, jitter).
*   Packets captured using **tshark** and **rtpengine**.

### **QoE Analysis**

*   RTP streams stored and analyzed using **ViSQOL**.
*   **MOS (Mean Opinion Score)** calculated to assess impact of impairments on call quality.

***

## **Results**

*   Internet bearer worked smoothly via gNB.
*   VoIP calls successfully tested with QoE metrics.
*   MOS dropped too quicly indicating sensitivity of Voice in 5G
*   Demonstrated feasibility of impairment analysis on 5G VoIP traffic.

***

## **Future Work**

*   Complete IMS registration for true VoNR functionality.
*   Extend testbed for real IMS-based voice calls and QoE evaluation.
*   Reduce the increments of the impairments

***

### **Architecture Diagram**
Network_Topology.png
**


✍️ *Author: Rasekoai Mokose* 
  📅 *Final Year Project — 2025*
  * Supervisor: Associate Professor J. Mwangama 

