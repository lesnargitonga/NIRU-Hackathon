from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import BaseDocTemplate, Paragraph, Spacer, PageBreak, Frame, PageTemplate, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas
from datetime import datetime

class RoadmapPDF:
    def __init__(self, filename):
        self.filename = filename
        self.doc = BaseDocTemplate(filename, pagesize=LETTER,
                                   rightMargin=50, leftMargin=50,
                                   topMargin=50, bottomMargin=50)
        self.styles = getSampleStyleSheet()
        self.create_styles()
        self.elements = []

    def create_styles(self):
        # Professional Color Palette
        self.primary_color = colors.HexColor("#003366")  # Navy Blue
        self.secondary_color = colors.HexColor("#800000")  # Maroon/Dark Red
        self.text_color = colors.HexColor("#2C3E50")     # Dark Slate Grey
        self.light_bg = colors.HexColor("#F0F8FF")       # Alice Blue

        self.styles.add(ParagraphStyle(
            name='RoadmapTitle',
            parent=self.styles['Heading1'],
            fontSize=26,
            leading=32,
            alignment=TA_CENTER,
            textColor=self.primary_color,
            spaceAfter=10,
            fontName="Helvetica-Bold"
        ))
        
        self.styles.add(ParagraphStyle(
            name='RoadmapSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceAfter=20,
            fontName="Helvetica"
        ))

        self.styles.add(ParagraphStyle(
            name='PhaseTitle',
            parent=self.styles['Heading2'],
            fontSize=18,
            leading=22,
            textColor=colors.white,
            backColor=self.primary_color,
            borderPadding=(10, 5, 10, 5), # Top, Right, Bottom, Left
            spaceBefore=20,
            spaceAfter=10,
            keepWithNext=True,
            fontName="Helvetica-Bold",
            borderRadius=5
        ))

        self.styles.add(ParagraphStyle(
            name='DateHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            leading=16,
            textColor=self.secondary_color,
            spaceBefore=12,
            spaceAfter=4,
            fontName="Helvetica-Bold"
        ))

        self.styles.add(ParagraphStyle(
            name='RoadmapBody',  # Renamed to avoid key collision
            parent=self.styles['Normal'],
            fontSize=12,
            leading=15,
            textColor=self.text_color,
            alignment=TA_LEFT,
            fontName="Helvetica"
        ))

        self.styles.add(ParagraphStyle(
            name='RoadmapBullet', # Renamed to avoid key collision
            parent=self.styles['Normal'],
            fontSize=12,
            leading=15,
            textColor=self.text_color,
            leftIndent=20,
            firstLineIndent=0,
            spaceAfter=4,
            fontName="Helvetica"
        ))
        
        self.styles.add(ParagraphStyle(
            name='MissionBox',
            parent=self.styles['Normal'],
            fontSize=12,
            leading=16,
            textColor=colors.black,
            backColor=colors.lightgrey,
            borderPadding=10,
            alignment=TA_JUSTIFY,
            fontName="Helvetica-Oblique",
            leftIndent=10,
            rightIndent=10
        ))

    def header_footer(self, canvas, doc):
        canvas.saveState()
        
        # Header
        canvas.setFont('Helvetica-Bold', 10)
        canvas.setFillColor(colors.grey)
        canvas.drawString(inch, LETTER[1] - 0.5 * inch, "OPERATION SENTINEL // TECH ROADMAP")
        canvas.drawRightString(LETTER[0] - inch, LETTER[1] - 0.5 * inch, datetime.now().strftime("%Y-%m-%d"))
        canvas.line(inch, LETTER[1] - 0.55 * inch, LETTER[0] - inch, LETTER[1] - 0.55 * inch)

        # Footer
        canvas.setFont('Helvetica', 9)
        canvas.drawString(inch, 0.5 * inch, "CONFIDENTIAL // INTERNAL USE ONLY")
        canvas.drawRightString(LETTER[0] - inch, 0.5 * inch, f"Page {doc.page}")
        canvas.line(inch, 0.6 * inch, LETTER[0] - inch, 0.6 * inch)
        
        canvas.restoreState()

    def add_content(self):
        # Title Section
        self.elements.append(Paragraph("OPERATION SENTINEL", self.styles['RoadmapTitle']))
        self.elements.append(Paragraph("Autonomous Perimeter Defense System", self.styles['RoadmapSubtitle']))
        
        # Mission Statement Box
        mission_text = "<b>MISSION OBJECTIVE:</b> Engineer a deployable, National Security-grade autonomous surveillance asset. This roadmap details the critical path from initial architecture to a fully sovereign, privacy-compliant unit capable of operating in GPS-denied environments."
        self.elements.append(Paragraph(mission_text, self.styles['MissionBox']))
        self.elements.append(Spacer(1, 20))

        # Phase 1
        self.elements.append(Paragraph("PHASE 1: THE CORE FOUNDATION (Immediate - Jan 30)", self.styles['PhaseTitle']))
        self.elements.append(Paragraph("<b>GOAL:</b> Validate Core Architecture (PX4-SITL Bridge) & System Design.", self.styles['RoadmapBody']))
        
        self.elements.append(Paragraph("Milestone 1: API Bridge & Protocol Validation", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>Action:</b> Verify bi-directional MAVLink communication between Python Mission Control (AirSim API) and PX4 Flight Stack.", self.styles['RoadmapBullet']))
        self.elements.append(Paragraph("• <b>Metric:</b> Achieve stable command stream with <10ms round-trip latency.", self.styles['RoadmapBullet']))

        self.elements.append(Paragraph("Milestone 2: System Architecture Blueprinting", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>Action:</b> Define segregation of concerns: Real-time Flight Controller vs. High-Level AI Planner.", self.styles['RoadmapBullet']))
        self.elements.append(Paragraph("• <b>Deliverable:</b> High-fidelity Architecture Diagram (ROS 2 / MAVROS node graph).", self.styles['RoadmapBullet']))

        self.elements.append(Paragraph("Milestone 3: Compliance Logic & Synthetic Data", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>Action:</b> Implement strict privacy-safe constraints. Use Synthetic Training Environment (STE) to eliminate PII risks.", self.styles['RoadmapBullet']))
        
        self.elements.append(Paragraph("Milestone 4: Dynamics Verification (Proof of Flight)", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>Action:</b> Execute initial PID tuning. Verify 'Hold Position' stability under zero-wind.", self.styles['RoadmapBullet']))
        self.elements.append(Paragraph("• <b>Success Criteria:</b> <5cm position drift over 60 seconds.", self.styles['RoadmapBullet']))
        
        self.elements.append(Paragraph("Final Step: SUBMIT TECHNICAL ROADMAP", self.styles['DateHeader']))

        # Phase 2
        self.elements.append(Paragraph("PHASE 2: THE DATA ENGINE (Jan 31 – Feb 13)", self.styles['PhaseTitle']))
        self.elements.append(Paragraph("<b>GOAL:</b> Scalable Data Pipeline & Procedural Environment Generation.", self.styles['RoadmapBody']))

        self.elements.append(Paragraph("Feb 1 - Feb 5: Procedural Stress Testing", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>Implementation:</b> Script random map generator (Perlin Noise) to spawn dense obstacle fields.", self.styles['RoadmapBullet']))
        self.elements.append(Paragraph("• <b>Scale:</b> Execute overnight headless simulation runs. Accumulate 10,000+ collision events.", self.styles['RoadmapBullet']))

        self.elements.append(Paragraph("Feb 6 - Feb 10: High-Frequency Telemetry Pipeline", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>Implementation:</b> Configure PX4 ULog streaming to local time-series DB (PostgreSQL).", self.styles['RoadmapBullet']))
        self.elements.append(Paragraph("• <b>Auditability:</b> Capture IMU, LiDAR, and Control Actuator outputs at 50Hz for forensics.", self.styles['RoadmapBullet']))

        self.elements.append(Paragraph("Feb 11 - Feb 13: LiDAR-Based Navigation", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>Demo:</b> Proof of concept flight relying exclusively on Sparse LiDAR (VFH+). No cameras.", self.styles['RoadmapBullet']))
        self.elements.append(Paragraph("• <b>Milestone:</b> 'Environment & Data' Submission.", self.styles['RoadmapBullet']))
        
        self.elements.append(PageBreak()) # Clean break for readability

        # Phase 3
        self.elements.append(Paragraph("PHASE 3: FUNCTIONAL ALPHA (Feb 14 – Feb 27)", self.styles['PhaseTitle']))
        self.elements.append(Paragraph("<b>GOAL:</b> Deploy Robust Visuomotor Policy (MVP).", self.styles['RoadmapBody']))

        self.elements.append(Paragraph("Feb 14 - Feb 20: Training the Recurrent Policy", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>ML Architecture:</b> Train PPO agent with LSTM memory cells for state estimation.", self.styles['RoadmapBullet']))
        self.elements.append(Paragraph("• <b>Dataset:</b> Utilize 100k+ step synthetic dataset for Behavior Cloning warm-starting.", self.styles['RoadmapBullet']))

        self.elements.append(Paragraph("Feb 21 - Feb 25: Disturbance Rejection", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>Test Setup:</b> Inject randomized wind vectors (up to 12 m/s) in Gazebo.", self.styles['RoadmapBullet']))
        self.elements.append(Paragraph("• <b>Criterion:</b> Maintain trajectory within 0.5m corridor.", self.styles['RoadmapBullet']))

        self.elements.append(Paragraph("Feb 26 - Feb 27: Containerization", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>DevOps:</b> Encapsulate Simulation + Stack into monolithic Docker container.", self.styles['RoadmapBullet']))
        self.elements.append(Paragraph("• <b>Milestone:</b> 'Functional Alpha' Submission.", self.styles['RoadmapBullet']))

        # Phase 4
        self.elements.append(Paragraph("PHASE 4: SECURITY & ASSURANCE (Feb 28 – Mar 13)", self.styles['PhaseTitle']))
        self.elements.append(Paragraph("<b>GOAL:</b> Safety Critical Systems & Human-in-the-Loop.", self.styles['RoadmapBody']))

        self.elements.append(Paragraph("Feb 28 - Mar 5: Hardware Kill Switch", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>Mechanism:</b> High-priority interrupt thread monitoring physical input.", self.styles['RoadmapBullet']))
        self.elements.append(Paragraph("• <b>Latency:</b> Override engagement <50ms.", self.styles['RoadmapBullet']))

        self.elements.append(Paragraph("Mar 6 - Mar 10: Deterministic Geofencing", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>Logic:</b> Hard-coded geometric geofence enforcement overriding Neural Net.", self.styles['RoadmapBullet']))
        
        self.elements.append(Paragraph("Mar 11 - Mar 13: C2 Dashboard Integration", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>Stack:</b> React/Flask WebSocket link for real-time threat maps.", self.styles['RoadmapBullet']))
        self.elements.append(Paragraph("• <b>Milestone:</b> 'System Security' Submission.", self.styles['RoadmapBullet']))

        # Phase 5
        self.elements.append(Paragraph("PHASE 5: THE X-FACTOR (Mar 14 – Mar 20)", self.styles['PhaseTitle']))
        self.elements.append(Paragraph("<b>GOAL:</b> Sovereign Deployment & Production Pitch.", self.styles['RoadmapBody']))

        self.elements.append(Paragraph("Mar 14 - Mar 17: Sovereignty Audit", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>Claim:</b> 100% Local Compute. Zero cloud dependency. Air-gappable.", self.styles['RoadmapBullet']))

        self.elements.append(Paragraph("Mar 18 - Mar 19: The 'Golden Run'", self.styles['DateHeader']))
        self.elements.append(Paragraph("• <b>Production:</b> 4K automated flight logs in high-complexity environments.", self.styles['RoadmapBullet']))
        self.elements.append(Paragraph("• <b>HUD Overlay:</b> Post-process video with telemetry overlays.", self.styles['RoadmapBullet']))

        self.elements.append(Paragraph("Mar 20: FINAL PITCH", self.styles['DateHeader']))


    def build(self):
        self.add_content()
        frame = Frame(self.doc.leftMargin, self.doc.bottomMargin, self.doc.width, self.doc.height, id='normal')
        template = PageTemplate(id='roadmaptemplate', frames=frame, onPage=self.header_footer)
        self.doc.addPageTemplates([template])
        self.doc.build(self.elements)

if __name__ == "__main__":
    pdf = RoadmapPDF("Operation_Sentinel_Roadmap.pdf")
    pdf.build()
    print("PDF Generated successfully: Operation_Sentinel_Roadmap.pdf")
