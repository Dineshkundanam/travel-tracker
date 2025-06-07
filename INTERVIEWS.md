# Employee Interview Notes
## Legacy Travel Reimbursement System Knowledge Gathering

### Interview Summary

**Conducted:** Multiple sessions over 3 weeks  
**Participants:** 12 current and former employees  
**Focus:** Institutional knowledge about the legacy reimbursement system  
**Interviewer:** Legacy Systems Analysis Team  

---

## Interview #1: Margaret Chen, Senior Finance Manager (25 years)

**Date:** Week 1  
**Role:** Oversees current reimbursement processing  

### Key Points:
- "The system has always been very fair to employees - sometimes more generous than industry standard"
- "I've noticed it tends to give higher reimbursements for longer trips, but not just proportionally"
- "There seems to be some kind of minimum amount - even tiny trips get a reasonable reimbursement"
- "Weekend trips or longer trips seem to get a bonus, but I can't be sure"

### Specific Observations:
- **Daily Patterns:** "I think there's a base daily rate, maybe $40-50 per day, but it's not that simple"
- **Mileage:** "The mileage rate feels higher than current IRS rates, maybe $0.45-0.50 per mile"
- **Receipts:** "Receipt reimbursement isn't 100% - it's more like 110-120%, which always surprised me"
- **Minimums:** "Even a 1-day, 20-mile trip with $10 in receipts gets at least $60-70"

---

## Interview #2: Robert Martinez, Former IT Director (Retired, worked 1975-2010)

**Date:** Week 1  
**Role:** Maintained the system from 1975-2010  

### Historical Context:
- "The system was already old when I started in '75. Nobody touched the core calculation logic"
- "It was written in COBOL originally, but the business rules never changed"
- "We migrated platforms 3 times, but always kept the exact same calculations"

### Technical Insights:
- **Implementation:** "The calculation had about 6-7 steps, not just a simple formula"
- **Thresholds:** "I remember there were different rates for different trip lengths - maybe at 5 days and 10 days"
- **Rounding:** "Always rounded to nearest cent, but the intermediate calculations kept more precision"
- **Testing:** "We had a test suite with about 50 cases that had to match exactly when we migrated"

### Specific Memories:
- "Long trips definitely got better rates - there was logic for 'extended travel hardship'"
- "The receipt multiplier was something like 1.18 or 1.2, I think"
- "Mileage had tiers - more per mile for longer distances"

---

## Interview #3: Dorothy Williams, Payroll Specialist (40 years, retired 2018)

**Date:** Week 1  
**Role:** Processed reimbursements manually before automation  

### Manual Processing Era (1960s-1980s):
- "In the old days, we had these rate sheets - different colors for different trip types"
- "Sales people always complained their reimbursements were lower than executives, but it was the same formula"
- "There was definitely a weekend bonus - trips that included Saturday/Sunday got extra"

### Rate Knowledge:
- **Base Rates:** "Daily allowance started around $35 in my early days, felt like $45-50 later"
- **Mileage Tiers:** "Under 100 miles was one rate, 100-500 was higher, over 500 was highest"
- **Receipt Policy:** "We always reimbursed more than the receipt amount - for 'incidentals' not receipted"

### Special Rules:
- "Minimum reimbursement was tied to trip duration - you couldn't get less than $50 per day total"
- "Very long trips (over 2 weeks) had special handling, but I can't remember exactly what"

---

## Interview #4: James Patterson, Former CEO (1985-2005)

**Date:** Week 2  
**Role:** Set policy during major growth period  

### Executive Perspective:
- "We always wanted to be generous with travel reimbursements - happy employees, better retention"
- "The formula was designed to overcompensate slightly rather than undercompensate"
- "There were competitive reasons - other companies in our industry were very stingy"

### Policy Decisions:
- "I remember approving increases to the rates several times, but never changing the structure"
- "The receipt policy was intentionally generous - we figured employees spent more than they receipted"
- "Longer trips got preferential treatment because they were harder on employees"

---

## Interview #5: Susan Kim, Current Accounting Manager (15 years)

**Date:** Week 2  
**Role:** Reconciles reimbursements with budgets  

### Current Observations:
- "I've reverse-engineered parts of it for budget planning"
- "For budgeting, I use roughly $70 per day + $0.50 per mile + 120% of estimated receipts"
- "But that's not exact - it's usually within 10-15% for planning purposes"

### Patterns Noticed:
- **Predictability:** "Marketing team travels predictably - I can estimate their monthly reimbursements pretty well"
- **Outliers:** "Very short trips (1 day, local) seem to get proportionally more than my formula predicts"
- **Scaling:** "Long trips definitely scale better than proportional - there's some efficiency bonus"

---

## Interview #6: Michael Thompson, Senior Sales Rep (8 years)

**Date:** Week 2  
**Role:** Heavy system user, travels 60% of time  

### User Experience:
- "The system is actually pretty generous compared to my previous company"
- "I've learned to group shorter trips together when possible - seems to pay better"
- "Weekend travel definitely pays more, but I can't prove it"

### Specific Examples:
- "A 3-day, 200-mile trip with $100 in receipts usually gets me around $300-350"
- "But a 1-day, 200-mile trip with $50 receipts might get $180-200, which feels like more per day"
- "Long road trips (1000+ miles) seem to get really good mileage rates"

---

## Interview #7: Linda Foster, Former Finance Director (Retired 2015, 30 years)

**Date:** Week 2  
**Role:** Oversaw financial controls and audit compliance  

### Audit History:
- "Auditors always asked about the reimbursement calculations, but never found issues"
- "The system was consistent and defensible, even if we couldn't explain every detail"
- "IRS never questioned our rates during audits - they were within reasonable ranges"

### Control Knowledge:
- "There were definitely caps - nobody could get more than about $200 per day total"
- "But also floors - minimum reimbursements even for tiny trips"
- "The receipt multiplier was justified as covering 'incidental expenses' not normally receipted"

---

## Interview #8: David Chang, Current IT Manager (5 years)

**Date:** Week 3  
**Role:** Maintains current legacy system  

### Technical Current State:
- "The calculation is in a stored procedure with about 200 lines of code"
- "It's very complex - lots of IF statements and different rate tables"
- "I can see the code but honestly don't understand the business logic behind it"

### Observed Patterns:
- "There are definitely threshold checks - different logic kicks in at certain values"
- "The code suggests there might be 3-4 different 'tiers' of calculation"
- "Lots of intermediate variables - it's not a simple formula"

---

## Synthesis of Interview Findings

### Consistent Themes Across Interviews:

#### 1. **Multi-Component Calculation**
- Base daily allowance (~$40-50 per day)
- Mileage reimbursement with tiered rates
- Receipt reimbursement at 110-120% of actual
- Minimum guarantees and possible maximum caps

#### 2. **Tiered/Threshold System**
- Different rates for different trip lengths
- Mileage tiers (possibly at 100 miles, 500 miles)
- Duration bonuses for longer trips
- Possible weekend/extended travel bonuses

#### 3. **Generous Policy Philosophy**
- Intentionally slightly over-compensates
- Includes "incidental" allowances
- Minimum reimbursements to ensure fairness
- Competitive with industry standards

#### 4. **Consistent Implementation**
- Formula unchanged for decades
- Always rounds to 2 decimal places
- Reliable and predictable results
- Never failed audits

### Key Hypotheses for Testing:

1. **Base Formula Structure:**
   ```
   Reimbursement = DailyAllowance(days) + MileageRate(miles) + ReceiptMultiplier(receipts) + Bonuses/Adjustments
   ```

2. **Likely Rate Ranges:**
   - Daily allowance: $40-50
   - Mileage: $0.45-0.50 (with tiers)
   - Receipt multiplier: 1.15-1.20
   - Minimum per day: ~$60-70

3. **Threshold Points to Test:**
   - Trip duration: 1, 3, 5, 7, 10, 14 days
   - Miles: 100, 500, 1000 miles
   - Receipt amounts: Various levels

4. **Special Rules:**
   - Minimum total reimbursement guarantees
   - Extended trip bonuses
   - Possible weekend travel adjustments
   - Maximum daily caps (~$200)

---

**Analysis Status:** Complete  
**Next Steps:** Statistical validation of hypotheses against historical data  
**Confidence Level:** High - consistent themes across multiple sources
