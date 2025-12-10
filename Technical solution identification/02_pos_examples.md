Example 1:
Input:
Problems: 0: Nut loosening | 1: Assembly complexity
Patent: "The fastener includes a nylon insert ring embedded within the internal threads. This insert creates frictional interference with the mating screw, eliminating the need for lock washers."

Reasoning:
1. Solution: "Nylon insert ring embedded in internal threads"
2. Type: Physical component → "Apparatus"
3. Problems: Frictional interference prevents loosening (0), eliminates lock washers (1)
4. Description: Nylon deforms against threads, creating friction to prevent rotation without extra parts.

Output:
{
  "solutions": [
    {
      "solution": "Nylon insert ring embedded in internal threads",
      "solution_type": "Apparatus",
      "related_problem_ids": [0, 1],
      "description": "The nylon ring creates frictional interference with mating threads, preventing loosening under vibration and eliminating the need for lock washers."
    }
  ]
}

Example 2:
Input:
Problems: 5: High power consumption | 8: Screen overheating
Patent: "The display controller implements a dynamic dimming module that adjusts the backlight duty cycle based on the average picture level (APL). By reducing current during dark scenes, thermal generation is minimized."

Reasoning:
1. Solution: "Dynamic dimming module based on APL"
2. Type: Integrated controller + sensor + logic → "System"
3. Problems: Reduced current lowers power (5) and heat (8)
4. Description: Analyzes video brightness to adjust backlight power dynamically.

Output:
{
  "solutions": [
    {
      "solution": "Dynamic dimming module based on Average Picture Level",
      "solution_type": "System",
      "related_problem_ids": [5, 8],
      "description": "The module analyzes video brightness and reduces backlight duty cycle during dark scenes, directly decreasing power consumption and thermal generation."
    }
  ]
}

Example 3:
Input:
Problems: 0: Prolonged refueling time | 13: Temperature overshoot
Patent: "The system uses a predictive algorithm to terminate refueling at an upper state of charge based on pre-estimated correction values, compensating for pressure drop after valve closure."

Reasoning:
1. Solution: "Terminate refueling at pre-calculated upper charge level"
2. Type: Algorithmic procedure → "Method"
3. Problems: Single-cycle filling reduces time (0), compensation prevents overshoot (13)
4. Description: Fills above 100% using correction values for post-closure effects.

Output:
{
  "solutions": [
    {
      "solution": "Terminate refueling at pre-calculated upper charge level",
      "solution_type": "Method",
      "related_problem_ids": [0, 13],
      "description": "The algorithm fills above rated density using pre-estimated correction values for pressure drop and thermal stabilization, achieving target charge in one cycle without temperature violations."
    }
  ]
}
