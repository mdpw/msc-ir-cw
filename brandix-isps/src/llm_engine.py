"""
LLM Engine - Ollama/Llama 3.1 Integration
Generates intelligent improvement suggestions
"""

import requests
import json
from typing import Dict, List

class LLMEngine:
    def __init__(self, model_name="phi3:mini", base_url="http://localhost:11434"):
        """Initialize Ollama LLM connection"""
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def test_connection(self):
        """Test if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                print("✓ Ollama connection successful")
                models = response.json().get('models', [])
                print(f"  Available models: {[m['name'] for m in models]}")
                return True
            else:
                print("❌ Ollama not responding correctly")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("  Make sure Ollama is running: ollama serve")
            return False
    
    def generate(self, prompt: str, max_tokens=500) -> str:
        """Generate text using Ollama"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"Error: {response.status_code} - {response.text}"
        
        except Exception as e:
            return f"Error generating response: {e}"
    
    def generate_improvement_suggestions(self, gap_objective: Dict, top_actions: List[Dict]) -> Dict:
        """
        Generate intelligent improvement suggestions for a gap objective
        
        Args:
            gap_objective: Dict with objective details and alignment score
            top_actions: List of top matching actions (even if weak)
        
        Returns:
            Dictionary with categorized suggestions
        """
        
        # Build context
        obj_text = gap_objective['objective']
        pillar = gap_objective.get('pillar', 'Unknown')
        score = gap_objective['alignment_score']
        
        actions_context = "\n".join([
            f"- {a.get('title', a.get('text', 'N/A'))} (Score: {a.get('similarity', 0):.1%})"
            for a in top_actions[:5]
        ])
        
        # Comprehensive prompt
        prompt = f"""You are a strategic planning expert analyzing Brandix's strategic and action plans.

STRATEGIC OBJECTIVE (Weak Alignment - {score:.1f}%):
Pillar: {pillar}
Objective: {obj_text}

CURRENT BEST MATCHING ACTIONS:
{actions_context}

ANALYSIS TASK:
This strategic objective has weak alignment ({score:.1f}%) with current action items. Generate specific, actionable improvement suggestions in the following categories:

1. NEW ACTIONS: Suggest 2-3 specific new action items that directly address this objective
2. KPI ENHANCEMENTS: Recommend 2-3 measurable KPIs to track progress
3. TIMELINE ADJUSTMENTS: Suggest realistic timeline milestones (quarterly)
4. RESOURCE ALLOCATION: Identify key resources, budget, or team needs
5. RISK MITIGATION: Highlight 1-2 potential risks and mitigation strategies

Format your response as:

**NEW ACTIONS:**
- [Specific action 1]
- [Specific action 2]
- [Specific action 3]

**KPI ENHANCEMENTS:**
- [Measurable KPI 1]
- [Measurable KPI 2]

**TIMELINE ADJUSTMENTS:**
- [Q1-Q4 milestones]

**RESOURCE ALLOCATION:**
- [Budget/team/infrastructure needs]

**RISK MITIGATION:**
- [Risk 1 and mitigation]
- [Risk 2 and mitigation]

Be specific, actionable, and realistic for an apparel manufacturing company."""

        print(f"\n🤖 Generating suggestions for: {obj_text[:60]}...")
        
        # Generate response
        response = self.generate(prompt, max_tokens=800)
        
        # Parse response into structured format
        suggestions = self._parse_suggestions(response)
        
        return {
            'objective_id': gap_objective.get('objective_id', 'Unknown'),
            'objective': obj_text,
            'pillar': pillar,
            'current_score': score,
            'suggestions': suggestions,
            'raw_response': response
        }
    
    def _parse_suggestions(self, response: str) -> Dict:
        """Parse LLM response into structured categories"""
        categories = {
            'new_actions': [],
            'kpi_enhancements': [],
            'timeline_adjustments': [],
            'resource_allocation': [],
            'risk_mitigation': []
        }
        
        current_category = None
        
        for line in response.split('\n'):
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            # Detect category headers
            if 'NEW ACTIONS' in line.upper():
                current_category = 'new_actions'
            elif 'KPI' in line.upper():
                current_category = 'kpi_enhancements'
            elif 'TIMELINE' in line.upper():
                current_category = 'timeline_adjustments'
            elif 'RESOURCE' in line.upper():
                current_category = 'resource_allocation'
            elif 'RISK' in line.upper():
                current_category = 'risk_mitigation'
            elif line.startswith('-') or line.startswith('•'):
                # Add bullet point to current category
                if current_category:
                    text = line.lstrip('-•').strip()
                    if text:
                        categories[current_category].append(text)
        
        return categories
    
    def generate_executive_summary(self, overall_alignment: Dict, gaps: Dict) -> str:
        """Generate executive summary of synchronization analysis"""
        
        prompt = f"""Generate a concise executive summary of this strategic plan synchronization analysis:

OVERALL ALIGNMENT: {overall_alignment['overall_score']:.1f}%
CLASSIFICATION: {overall_alignment['classification']}
COVERAGE RATE: {overall_alignment['coverage_rate']:.1f}%

DISTRIBUTION:
- Strong alignments (≥70%): {overall_alignment['distribution']['strong']}
- Moderate alignments (50-70%): {overall_alignment['distribution']['moderate']}
- Weak alignments (<50%): {overall_alignment['distribution']['weak']}

GAPS IDENTIFIED:
- Weak objectives: {len(gaps['weak_objectives'])}
- Orphan actions: {len(gaps['orphan_actions'])}
- Weak pillars: {len(gaps['pillar_gaps'])}

Write a 3-paragraph executive summary (max 200 words) covering:
1. Overall synchronization status and key strengths
2. Critical gaps and areas requiring attention
3. High-level recommendations for leadership

Be professional, data-driven, and actionable."""

        print("\n📊 Generating executive summary...")
        return self.generate(prompt, max_tokens=400)


# Test
if __name__ == "__main__":
    print("="*80)
    print("LLM ENGINE TEST")
    print("="*80)
    
    llm = LLMEngine()
    
    # Test connection
    if not llm.test_connection():
        print("\n⚠️ Please start Ollama:")
        print("  1. Open terminal")
        print("  2. Run: ollama serve")
        print("  3. In another terminal: ollama pull llama3.1")
        exit(1)
    
    # Test basic generation
    print("\n" + "="*80)
    print("TEST 1: Basic Generation")
    print("="*80)
    response = llm.generate("What are the key benefits of renewable energy in manufacturing?", max_tokens=100)
    print(f"\nResponse: {response}")
    
    # Test improvement suggestions
    print("\n" + "="*80)
    print("TEST 2: Improvement Suggestions")
    print("="*80)
    
    test_gap = {
        'objective_id': 'OBJ-001',
        'objective': 'Achieve 100% renewable energy by 2030',
        'pillar': 'Environmental Leadership',
        'alignment_score': 45.2
    }
    
    test_actions = [
        {'title': 'Solar PV Installation Phase 1', 'similarity': 0.42},
        {'title': 'Energy Efficiency Program', 'similarity': 0.38}
    ]
    
    suggestions = llm.generate_improvement_suggestions(test_gap, test_actions)
    
    print(f"\n✓ Generated suggestions for: {test_gap['objective']}")
    print(f"\nNEW ACTIONS ({len(suggestions['suggestions']['new_actions'])}):")
    for action in suggestions['suggestions']['new_actions'][:3]:
        print(f"  - {action}")
    
    print("\n✓ Day 5-6 Complete! LLM Engine working!")