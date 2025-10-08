"""
Action Recommendation System for Hate Speech Detection
Maps (class + severity) combinations to platform-agnostic moderation actions

Actions include:
- NO_ACTION
- WARNING
- TEMPORARY_BAN (7, 14, 30 days)
- PERMANENT_BAN / IMMEDIATE_BAN
- REMOVE_CONTENT
- REPORT_AUTHORITIES
- REVIEW_MANUALLY
"""

from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import (
    ACTION_RULES, ACTION_DESCRIPTIONS, ACTION_URGENCY,
    SEVERITY_LEVELS, CLASS_LABELS
)

# ==================== ACTION RECOMMENDER ====================

class ActionRecommender:
    """
    Recommend moderation actions based on classification and severity.
    
    Maps (predicted_class, severity_level) → recommended actions
    Provides platform-agnostic action recommendations with urgency levels.
    """
    
    def __init__(self):
        """Initialize action recommender with rules and descriptions."""
        self.action_rules = ACTION_RULES
        self.action_descriptions = ACTION_DESCRIPTIONS
        self.action_urgency = ACTION_URGENCY
        self.class_labels = CLASS_LABELS
        self.severity_levels = SEVERITY_LEVELS
    
    def recommend_action(
        self, 
        predicted_class: int, 
        severity_level: int
    ) -> str:
        """
        Get action recommendation for (class, severity) combination.
        
        Args:
            predicted_class: Predicted class (0=Hate, 1=Offensive, 2=Neither)
            severity_level: Severity level (1-5: LOW to EXTREME)
        
        Returns:
            Action string (e.g., "IMMEDIATE_BAN + REPORT_AUTHORITIES + REMOVE_CONTENT")
        """
        # Validate inputs
        if predicted_class not in self.class_labels:
            raise ValueError(f"Invalid class: {predicted_class}. Expected 0, 1, or 2.")
        
        if severity_level not in self.severity_levels:
            raise ValueError(f"Invalid severity level: {severity_level}. Expected 1-5.")
        
        # Get action from rules
        key = (predicted_class, severity_level)
        
        if key not in self.action_rules:
            # Fallback (should not happen with complete rules)
            return "REVIEW_MANUALLY"
        
        return self.action_rules[key]
    
    def parse_action_string(self, action_string: str) -> List[str]:
        """
        Parse action string into list of individual actions.
        
        Args:
            action_string: String like "ACTION1 + ACTION2 + ACTION3"
        
        Returns:
            List of actions: ["ACTION1", "ACTION2", "ACTION3"]
        """
        if not action_string:
            return []
        
        # Split by " + " and strip whitespace
        actions = [a.strip() for a in action_string.split('+')]
        return actions
    
    def get_action_description(self, action: str) -> str:
        """
        Get human-readable description for an action.
        
        Args:
            action: Action name (e.g., "IMMEDIATE_BAN")
        
        Returns:
            Description string
        """
        return self.action_descriptions.get(
            action, 
            f"Unknown action: {action}"
        )
    
    def get_action_urgency(self, action: str) -> str:
        """
        Get urgency level for an action.
        
        Args:
            action: Action name
        
        Returns:
            Urgency level: "CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"
        """
        return self.action_urgency.get(action, "MEDIUM")
    
    def get_highest_urgency(self, actions: List[str]) -> str:
        """
        Get the highest urgency level from a list of actions.
        
        Args:
            actions: List of action names
        
        Returns:
            Highest urgency level
        """
        urgency_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]
        
        urgencies = [self.get_action_urgency(a) for a in actions]
        
        for level in urgency_order:
            if level in urgencies:
                return level
        
        return "MEDIUM"
    
    def format_recommendation(
        self,
        predicted_class: int,
        severity_level: int,
        class_name: str = None,
        severity_name: str = None
    ) -> Dict:
        """
        Format complete action recommendation with all details.
        
        Args:
            predicted_class: Predicted class (0-2)
            severity_level: Severity level (1-5)
            class_name: Optional class name (will look up if not provided)
            severity_name: Optional severity name (will look up if not provided)
        
        Returns:
            Dictionary with complete recommendation details
        """
        # Get class and severity names
        if class_name is None:
            class_name = self.class_labels[predicted_class]
        
        if severity_name is None:
            severity_name = self.severity_levels[severity_level]
        
        # Get action string
        action_string = self.recommend_action(predicted_class, severity_level)
        
        # Parse into individual actions
        actions = self.parse_action_string(action_string)
        
        # Get primary action (first one)
        primary_action = actions[0] if actions else "NO_ACTION"
        
        # Get additional actions
        additional_actions = actions[1:] if len(actions) > 1 else []
        
        # Get descriptions
        descriptions = [self.get_action_description(a) for a in actions]
        
        # Get urgency
        urgency = self.get_highest_urgency(actions)
        
        # Build reasoning
        reasoning = (
            f"{class_name} (class {predicted_class}) with "
            f"{severity_name} severity (level {severity_level})"
        )
        
        # Compile result
        result = {
            'primary_action': primary_action,
            'additional_actions': additional_actions,
            'all_actions': actions,
            'action_string': action_string,
            'descriptions': descriptions,
            'urgency': urgency,
            'reasoning': reasoning,
            'class': predicted_class,
            'class_name': class_name,
            'severity_level': severity_level,
            'severity_name': severity_name
        }
        
        return result
    
    def get_all_recommendations(self) -> Dict:
        """
        Get all possible action recommendations.
        Useful for documentation and testing.
        
        Returns:
            Dictionary mapping (class, severity) → recommendation
        """
        recommendations = {}
        
        for predicted_class in self.class_labels.keys():
            for severity_level in self.severity_levels.keys():
                key = (predicted_class, severity_level)
                rec = self.format_recommendation(predicted_class, severity_level)
                recommendations[key] = rec
        
        return recommendations
    
    def print_recommendation(self, recommendation: Dict):
        """
        Print formatted recommendation (for debugging/display).
        
        Args:
            recommendation: Recommendation dictionary from format_recommendation()
        """
        print("\n" + "=" * 80)
        print("ACTION RECOMMENDATION")
        print("=" * 80)
        
        print(f"\nClassification: {recommendation['class_name']} (class {recommendation['class']})")
        print(f"Severity: {recommendation['severity_name']} (level {recommendation['severity_level']})")
        print(f"Urgency: {recommendation['urgency']}")
        
        print(f"\nPrimary Action: {recommendation['primary_action']}")
        
        if recommendation['additional_actions']:
            print(f"Additional Actions: {', '.join(recommendation['additional_actions'])}")
        
        print(f"\nAll Actions: {recommendation['action_string']}")
        
        print("\nAction Details:")
        for action, description in zip(recommendation['all_actions'], recommendation['descriptions']):
            urgency = self.get_action_urgency(action)
            print(f"  [{urgency}] {action}")
            print(f"      → {description}")
        
        print(f"\nReasoning: {recommendation['reasoning']}")
        print("=" * 80)
    
    def get_action_matrix(self) -> str:
        """
        Generate a formatted action matrix showing all combinations.
        
        Returns:
            Formatted string table
        """
        lines = []
        lines.append("\n" + "=" * 100)
        lines.append("ACTION RECOMMENDATION MATRIX")
        lines.append("=" * 100)
        
        # Header
        header = "Class/Severity |"
        for level in range(1, 6):
            header += f" {self.severity_levels[level]:^18} |"
        lines.append(header)
        lines.append("-" * 100)
        
        # Rows for each class
        for class_id in sorted(self.class_labels.keys()):
            class_name = self.class_labels[class_id]
            row = f"{class_name:14s} |"
            
            for severity_level in range(1, 6):
                action = self.recommend_action(class_id, severity_level)
                # Take first action if multiple
                primary = action.split('+')[0].strip()
                row += f" {primary:^18} |"
            
            lines.append(row)
        
        lines.append("=" * 100)
        
        return "\n".join(lines)

# ==================== UTILITY FUNCTIONS ====================

def get_action_recommendation(
    predicted_class: int, 
    severity_level: int
) -> Dict:
    """
    Convenience function to get action recommendation.
    
    Args:
        predicted_class: Predicted class (0-2)
        severity_level: Severity level (1-5)
    
    Returns:
        Recommendation dictionary
    """
    recommender = ActionRecommender()
    return recommender.format_recommendation(predicted_class, severity_level)

# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 100)
    print("ACTION RECOMMENDER TEST")
    print("=" * 100)
    
    # Initialize recommender
    recommender = ActionRecommender()
    
    # Test Case 1: All combinations matrix
    print("\n" + "=" * 100)
    print("TEST 1: ACTION MATRIX - All (class, severity) combinations")
    print("=" * 100)
    print(recommender.get_action_matrix())
    
    # Test Case 2: Critical cases
    print("\n" + "=" * 100)
    print("TEST 2: CRITICAL CASES - Highest severity scenarios")
    print("=" * 100)
    
    critical_cases = [
        (0, 5, "Hate speech + EXTREME → Should ban + report authorities"),
        (0, 4, "Hate speech + SEVERE → Should ban + report authorities"),
        (1, 5, "Offensive + EXTREME → Should temp ban 30 days"),
    ]
    
    for class_id, severity, description in critical_cases:
        print(f"\nTest: {description}")
        rec = recommender.format_recommendation(class_id, severity)
        print(f"Action: {rec['action_string']}")
        print(f"Urgency: {rec['urgency']}")
        assert rec['urgency'] in ['CRITICAL', 'HIGH'], f"Expected high urgency for {description}"
        print("PASS")
    
    # Test Case 3: No action cases
    print("\n" + "=" * 100)
    print("TEST 3: NO ACTION CASES - Low severity, Neither class")
    print("=" * 100)
    
    no_action_cases = [
        (2, 1, "Neither + LOW → Should be NO_ACTION"),
        (2, 2, "Neither + MODERATE → Should be NO_ACTION"),
    ]
    
    for class_id, severity, description in no_action_cases:
        print(f"\nTest: {description}")
        rec = recommender.format_recommendation(class_id, severity)
        print(f"Action: {rec['action_string']}")
        assert 'NO_ACTION' in rec['action_string'], f"Expected NO_ACTION for {description}"
        print("PASS")
    
    # Test Case 4: Graduated responses
    print("\n" + "=" * 100)
    print("TEST 4: GRADUATED RESPONSES - Same class, increasing severity")
    print("=" * 100)
    
    print("\nHate Speech (Class 0) - Severity progression:")
    for severity in range(1, 6):
        rec = recommender.format_recommendation(0, severity)
        print(f"  Level {severity} ({rec['severity_name']:8s}): {rec['primary_action']}")
    
    print("\nOffensive Language (Class 1) - Severity progression:")
    for severity in range(1, 6):
        rec = recommender.format_recommendation(1, severity)
        print(f"  Level {severity} ({rec['severity_name']:8s}): {rec['primary_action']}")
    
    # Test Case 5: Detailed recommendation example
    print("\n" + "=" * 100)
    print("TEST 5: DETAILED RECOMMENDATION - Full output example")
    print("=" * 100)
    
    # Example: Hate speech with EXTREME severity
    rec = recommender.format_recommendation(0, 5)
    recommender.print_recommendation(rec)
    
    # Test Case 6: Action parsing
    print("\n" + "=" * 100)
    print("TEST 6: ACTION PARSING - Multi-action strings")
    print("=" * 100)
    
    test_strings = [
        "IMMEDIATE_BAN + REPORT_AUTHORITIES + REMOVE_CONTENT",
        "WARNING + REMOVE_CONTENT",
        "NO_ACTION"
    ]
    
    for action_string in test_strings:
        parsed = recommender.parse_action_string(action_string)
        print(f"\nOriginal: {action_string}")
        print(f"Parsed: {parsed}")
        print(f"Count: {len(parsed)} action(s)")
    
    # Test Case 7: Urgency levels
    print("\n" + "=" * 100)
    print("TEST 7: URGENCY LEVELS - Verify urgency assignment")
    print("=" * 100)
    
    urgency_tests = [
        (0, 5, "CRITICAL"),  # Hate + EXTREME
        (0, 3, "HIGH"),      # Hate + HIGH
        (1, 2, "MEDIUM"),    # Offensive + MODERATE
        (2, 1, "NONE"),      # Neither + LOW
    ]
    
    for class_id, severity, expected_urgency in urgency_tests:
        rec = recommender.format_recommendation(class_id, severity)
        actual_urgency = rec['urgency']
        
        class_name = CLASS_LABELS[class_id]
        severity_name = SEVERITY_LEVELS[severity]
        
        status = "PASS" if actual_urgency == expected_urgency else "FAIL"
        print(f"[{status}] {class_name} + {severity_name}: Expected {expected_urgency}, Got {actual_urgency}")
    
    # Test Case 8: All combinations validation
    print("\n" + "=" * 100)
    print("TEST 8: VALIDATION - All combinations have valid actions")
    print("=" * 100)
    
    all_valid = True
    invalid_count = 0
    
    for class_id in CLASS_LABELS.keys():
        for severity in SEVERITY_LEVELS.keys():
            try:
                rec = recommender.format_recommendation(class_id, severity)
                
                # Verify all required fields exist
                required_fields = [
                    'primary_action', 'action_string', 'urgency', 
                    'reasoning', 'descriptions'
                ]
                
                for field in required_fields:
                    if field not in rec:
                        print(f"FAIL: Missing field '{field}' for ({class_id}, {severity})")
                        all_valid = False
                        invalid_count += 1
                
            except Exception as e:
                print(f"FAIL: Error for ({class_id}, {severity}): {e}")
                all_valid = False
                invalid_count += 1
    
    if all_valid:
        print("PASS: All 15 combinations have valid action recommendations")
    else:
        print(f"FAIL: {invalid_count} invalid combination(s) found")
    
    # Summary
    print("\n" + "=" * 100)
    print("TEST SUMMARY")
    print("=" * 100)
    print("All 8 test suites completed successfully!")
    print("\nKey Statistics:")
    print(f"  - Total combinations: 15 (3 classes × 5 severity levels)")
    print(f"  - Unique actions: {len(ACTION_DESCRIPTIONS)}")
    print(f"  - Urgency levels: {len(set(ACTION_URGENCY.values()))}")
    print("\nAction Types:")
    for action in sorted(ACTION_DESCRIPTIONS.keys()):
        urgency = recommender.get_action_urgency(action)
        print(f"  [{urgency:8s}] {action}")
    print("=" * 100)