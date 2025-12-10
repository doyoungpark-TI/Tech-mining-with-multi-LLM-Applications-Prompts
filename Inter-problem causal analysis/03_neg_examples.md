[NEGATIVE GUIDELINES]
The following should NOT be extracted as causal relations:

1. Correlation without Causation
   Context: "The system exhibited both high vibration and loud noise during operation."
   Reason: Co-occurrence ("and") does not establish directional causality.
   Action: Return [].

2. Domain Knowledge without Textual Evidence
   Context: "Tank pressure measured at 700 bar."
   Symptoms: ["HighPressure", "ExplosionRisk"]
   Reason: No textual mention of risk or effect, only measurement data.
   Action: Return [].

3. Reverse Causality Error
   Context: "The overheating was triggered by the fan failure."
   Wrong: Overheating -> Fan failure
   Correct: Fan failure -> Overheating
   Reason: "X triggered by Y" means Y causes X.
   Action: Extract correct direction only.

4. Pure Temporal Sequence
   Context: "After bearing replacement, vibration continued."
   Reason: "After" shows time order, "continued" suggests pre-existing condition.
   Action: Return [].
