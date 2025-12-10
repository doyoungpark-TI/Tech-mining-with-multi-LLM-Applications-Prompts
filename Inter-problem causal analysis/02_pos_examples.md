positive_examples_data = """
Example 1: Explicit Causality (Direct Causal Verb)
Input:
Symptoms: [{"entity": "Valve.Leakage", "code": "V_0"}, {"entity": "System.PressureDrop", "code": "P_0"}]
Context: "The valve leakage caused a significant pressure drop in the system."

Reasoning:
1. Analyze Context: Direct causal statement with explicit verb.
2. Identify Connective: "caused" is a strong causal marker.
3. Determine Direction: Source = "Valve.Leakage" (V_0), Target = "System.PressureDrop" (P_0).
4. Assign Type & Confidence: Explicit marker present → type: "explicit", confidence: 0.9.

Expected output:
{
  "relations": [
    {
      "source": "Valve.Leakage",
      "source_code": "V_0",
      "target": "System.PressureDrop",
      "target_code": "P_0",
      "type": "explicit",
      "connective": "caused",
      "confidence": 0.9
    }
  ]
}


Example 2: Explicit Causality (Resultative Construction)
Input:
Symptoms: [{"entity": "Filter.Clogging", "code": "F_1"}, {"entity": "Pump.Overload", "code": "P_2"}]
Context: "Filter clogging resulted in pump overload during peak operation."

Reasoning:
1. Analyze Context: Sequential process with explicit outcome marker.
2. Identify Connective: "resulted in" links cause to effect directly.
3. Determine Direction: Source = "Filter.Clogging" (F_1), Target = "Pump.Overload" (P_2).
4. Assign Type & Confidence: Resultative construction present → type: "explicit", confidence: 0.85.

Expected output:
{
  "relations": [
    {
      "source": "Filter.Clogging",
      "source_code": "F_1",
      "target": "Pump.Overload",
      "target_code": "P_2",
      "type": "explicit",
      "connective": "resulted in",
      "confidence": 0.85
    }
  ]
}


Example 3: Implicit Causality (Conditional Dependency)
Input:
Symptoms: [{"entity": "Coolant.Depletion", "code": "C_0"}, {"entity": "Engine.Overheating", "code": "E_0"}]
Context: "When coolant levels dropped below minimum, the engine began overheating."

Reasoning:
1. Analyze Context: Conditional statement with temporal sequence.
2. Identify Connective: "When... began" shows temporal-causal relationship.
3. Determine Direction: Source = "Coolant.Depletion" (C_0), Target = "Engine.Overheating" (E_0).
4. Assign Type & Confidence: No explicit causal verb, but clear dependency → type: "implicit", confidence: 0.75.

Expected output:
{
  "relations": [
    {
      "source": "Coolant.Depletion",
      "source_code": "C_0",
      "target": "Engine.Overheating",
      "target_code": "E_0",
      "type": "implicit",
      "connective": "when... began",
      "confidence": 0.75
    }
  ]
}
"""
