"""
Executive Summary Generator
Uses LLM to generate professional executive summaries
Satisfies Requirement 3.5: LLM-based summarization and reporting
"""

import json
import os
from typing import Dict

class ExecutiveSummaryGenerator:
    def __init__(self, llm_engine):
        """Initialize with LLM engine"""
        self.llm = llm_engine
    
    def generate_executive_summary(self, sync_report: dict) -> dict:
        """
        Generate executive summary from synchronization report
        
        Args:
            sync_report: Complete synchronization report dictionary
            
        Returns:
            Dictionary with executive summary sections
        """
        print("\n" + "="*80)
        print("GENERATING EXECUTIVE SUMMARY")
        print("="*80)
        
        overall = sync_report['overall_alignment']
        gaps = sync_report['gaps']
        pillar_stats = sync_report['pillar_stats']
        
        # Build comprehensive context
        context = self._build_context(overall, gaps, pillar_stats)
        
        print("\n📄 Generating summary sections...")
        
        # Generate different sections
        summary = {
            'overview': self._generate_overview(context),
            'key_findings': self._generate_key_findings(context),
            'critical_gaps': self._generate_critical_gaps(context),
            'recommendations': self._generate_recommendations(context),
            'risk_assessment': self._generate_risk_assessment(context),
            'next_steps': self._generate_next_steps(context)
        }
        
        print("✓ All sections generated successfully!")
        
        return summary
    
    def _build_context(self, overall: dict, gaps: dict, pillar_stats: dict) -> str:
        """Build context string for LLM"""
        
        # Pillar performance summary
        pillar_summary = []
        for pillar, stats in pillar_stats.items():
            pillar_summary.append(
                f"- {pillar}: {stats['count']} objectives, "
                f"avg score {stats['average_score']:.1f}% ({stats['pillar_status']})"
            )
        
        context = f"""
SYNCHRONIZATION ANALYSIS RESULTS - BRANDIX STRATEGIC PLAN 2025-2030

OVERALL METRICS:
- Total Strategic Objectives: {overall['total_objectives']}
- Total Action Items: {overall['total_actions']}
- Overall Alignment Score: {overall['overall_score']:.1f}%
- Mean Max Similarity: {overall['mean_max_similarity']:.1f}%
- Classification: {overall['classification']}
- Coverage Rate: {overall['coverage_rate']:.1f}%

ALIGNMENT DISTRIBUTION:
- Strong alignments (≥70%): {overall['distribution']['strong']} objectives
- Moderate alignments (50-70%): {overall['distribution']['moderate']} objectives  
- Weak alignments (<50%): {overall['distribution']['weak']} objectives

GAPS IDENTIFIED:
- Weak objectives requiring attention: {len(gaps['weak_objectives'])}
- Orphan actions (unclear strategic link): {len(gaps['orphan_actions'])}
- Weak pillars: {len(gaps['pillar_gaps'])}
- Coverage gaps (no moderate/strong matches): {len(gaps['coverage_gaps'])}

PILLAR-LEVEL PERFORMANCE:
{chr(10).join(pillar_summary)}

CONTEXT:
This analysis assesses synchronization between Brandix's 5-year Strategic Plan (2025-2030)
and Year 1 Action Plan (2025-2026) for an apparel manufacturing company focused on
sustainability, innovation, and operational excellence.
"""
        return context
    
    def _generate_overview(self, context: str) -> str:
        """Generate executive overview (2-3 paragraphs)"""
        
        print("  1/6 Generating Executive Overview...")
        
        prompt = f"""{context}

Generate a concise executive overview (2-3 paragraphs, max 250 words) that:
1. Summarizes overall synchronization status in clear terms
2. Explains what the alignment score means for Brandix
3. Highlights the most important finding

Write in professional executive language. Be data-driven and objective.
Do not use bullet points - write in flowing paragraphs."""

        response = self.llm.generate(prompt, max_tokens=500)
        return response
    
    def _generate_key_findings(self, context: str) -> str:
        """Generate key findings section"""
        
        print("  2/6 Generating Key Findings...")
        
        prompt = f"""{context}

Generate 5-7 key findings from this analysis.

Focus on:
- Most significant alignment strengths
- Areas of concern that need attention
- Pillar-level insights
- Patterns across objectives

Use specific numbers and percentages. Be concise and impactful.
Format as a numbered list."""

        response = self.llm.generate(prompt, max_tokens=600)
        return response
    
    def _generate_critical_gaps(self, context: str) -> str:
        """Generate critical gaps analysis"""
        
        print("  3/6 Generating Critical Gaps...")
        
        prompt = f"""{context}

Identify the 3-5 most critical gaps requiring immediate attention.

For each gap:
- State the issue clearly
- Explain why it's critical for Brandix
- Suggest priority level (Critical/High/Medium)

Be specific about which strategic areas are under-supported.
Format as a numbered list."""

        response = self.llm.generate(prompt, max_tokens=600)
        return response
    
    def _generate_recommendations(self, context: str) -> str:
        """Generate strategic recommendations"""
        
        print("  4/6 Generating Strategic Recommendations...")
        
        prompt = f"""{context}

Provide 5-7 strategic recommendations for Brandix leadership.

Recommendations should:
- Be actionable and specific
- Address the identified gaps
- Leverage existing strengths
- Include suggested timeframes (Q1-Q4)

Format as a numbered list with clear action items."""

        response = self.llm.generate(prompt, max_tokens=700)
        return response
    
    def _generate_risk_assessment(self, context: str) -> str:
        """Generate risk assessment"""
        
        print("  5/6 Generating Risk Assessment...")
        
        prompt = f"""{context}

Assess strategic risks based on the alignment gaps.

Identify 3-4 key risks:
- Risk description
- Potential impact on business
- Likelihood (High/Medium/Low)
- Brief mitigation approach

Be specific to the weak areas identified in the data.
Format as a numbered list."""

        response = self.llm.generate(prompt, max_tokens=600)
        return response
    
    def _generate_next_steps(self, context: str) -> str:
        """Generate next steps"""
        
        print("  6/6 Generating Next Steps...")
        
        prompt = f"""{context}

Propose 5-7 immediate next steps (30-90 days) for the Brandix executive team.

Each step should:
- Be specific and actionable
- Include suggested timeline
- State expected outcome
- Prioritize based on gap severity

Format as a numbered list with clear actions and deadlines."""

        response = self.llm.generate(prompt, max_tokens=700)
        return response
    
    def save_summary(self, summary: dict, output_file: str):
        """Save executive summary to JSON"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Executive summary saved to {output_file}")
        return summary


# Test
if __name__ == "__main__":
    import sys
    sys.path.append('src')
    
    from llm_engine import LLMEngine
    
    print("="*80)
    print("EXECUTIVE SUMMARY GENERATOR TEST")
    print("="*80)
    
    # Load synchronization report
    print("\nStep 1: Loading synchronization report...")
    try:
        with open('outputs/synchronization_report.json', 'r', encoding='utf-8') as f:
            report = json.load(f)
        print(f"✓ Loaded report with {report['overall_alignment']['total_objectives']} objectives")
    except Exception as e:
        print(f"❌ Error loading report: {e}")
        print("Please run: python src/synchronization_engine.py")
        exit(1)
    
    # Initialize LLM
    print("\nStep 2: Initializing LLM (Phi-3 Mini)...")
    llm = LLMEngine(model_name="phi3:mini")
    if not llm.test_connection():
        print("❌ Ollama not running!")
        print("\nPlease start Ollama:")
        print("  1. Open terminal")
        print("  2. Run: ollama serve")
        exit(1)
    
    # Generate summary
    print("\nStep 3: Generating executive summary...")
    print("(This will take 2-3 minutes as LLM generates 6 sections)\n")
    
    generator = ExecutiveSummaryGenerator(llm)
    summary = generator.generate_executive_summary(report)
    
    # Display results
    print("\n" + "="*80)
    print("EXECUTIVE OVERVIEW")
    print("="*80)
    print(summary['overview'])
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(summary['key_findings'])
    
    print("\n" + "="*80)
    print("CRITICAL GAPS")
    print("="*80)
    print(summary['critical_gaps'])
    
    print("\n" + "="*80)
    print("STRATEGIC RECOMMENDATIONS")
    print("="*80)
    print(summary['recommendations'])
    
    print("\n" + "="*80)
    print("RISK ASSESSMENT")
    print("="*80)
    print(summary['risk_assessment'])
    
    print("\n" + "="*80)
    print("NEXT STEPS (30-90 Days)")
    print("="*80)
    print(summary['next_steps'])
    
    # Save
    generator.save_summary(summary, 'outputs/executive_summary.json')
    
    print("\n" + "="*80)
    print("✓ EXECUTIVE SUMMARY GENERATION COMPLETE!")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Review outputs/executive_summary.json")
    print("  2. Add to dashboard (update app.py)")
    print("  3. Test in Streamlit")