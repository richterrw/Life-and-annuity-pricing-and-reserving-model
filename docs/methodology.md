# Methodology

This document sets out the formulas behind each part of the model, and the code that implements each one. Notation follows *Actuarial Mathematics for Life Contingent Risks* (ALTAM). Time is measured in policy years, and $t$ counts completed years since issue.

**Contents**

1. [Mortality bases](#1-mortality-bases)
2. [Life-contingency functions](#2-life-contingency-functions)
3. [Statutory reserves](#3-statutory-reserves)
4. [Cash-flow projection and profit test](#4-cash-flow-projection-and-profit-test)
5. [Pricing](#5-pricing)
6. [Premium build-up](#6-premium-build-up)
7. [Deterministic sensitivities](#7-deterministic-sensitivities)
8. [Interest-rate scenarios](#8-interest-rate-scenarios)
9. [Stochastic reserve](#9-stochastic-reserve)
10. [Stochastic mortality](#10-stochastic-mortality)
11. [Simplifications](#11-simplifications)

---

## 1. Mortality bases

*Code: `lamodel/tables.py`*

Every basis is reduced to a vector of one-year death probabilities $q_{[x]+t}$, $t = 0, 1, \dots$, for a life issued at age $x$. The final rate is set to 1 so that the table closes.

| Basis | Construction |
|---|---|
| 2017 CSO select & ultimate | $q_{[x]+t}$ is read from the select table for durations 1–25, then $q_{x+t}$ from the ultimate table (SOA table 3291, nonsmoker male, ANB) |
| 2012 IAR (generational) | $q_x(y) = q^{2012}_x \,(1 - G2_x)^{\,y - 2012}$, where $y$ is the calendar year in which age $x$ is reached (SOA tables 2581 and 2583) |
| Scaled basis | $q' = \min(k\,q,\ 1)$, used for pricing mortality (70% of CSO for term) and for shocks |
| ALTAM SUSM | Makeham's law, $\mu_x = A + Bc^x$, integrated exactly: ${}_tp_x = \exp\!\big(-At - \tfrac{B}{\ln c}c^x(c^t - 1)\big)$ |

The XTbML files from mort.soa.org are parsed directly. Changing basis never changes product code.

## 2. Life-contingency functions

*Code: `lamodel/actuarial.py`*

All functions work in annual, curtate time: deaths are paid at the end of the year of death, and premiums and annuity payments at the start of the year. Interest may be a scalar or a vector of forward rates.

$$
{}_tp_x=\prod_{s<t}(1-q_{x+s}),\qquad
\ddot a_{x:\overline{n}|}=\sum_{t=0}^{n-1}v^t\,{}_tp_x,\qquad
A^1_{x:\overline{n}|}=\sum_{t=0}^{n-1}v^{t+1}\,{}_tp_x\,q_{x+t}
$$

$$
{}_{m|}\ddot a_x=\sum_{t\ge m}v^t\,{}_tp_x,\qquad
{}_nE_x=v^n\,{}_np_x,\qquad
A_{x:\overline{n}|}=1-d\,\ddot a_{x:\overline{n}|}\ \ \text{(tested)}
$$

**Validation.** On the SUSM at 5%, the model reproduces ALTAM's tabulated $l_x$, $\ddot a_x$ and $A_x$ at ages 20, 50, 60 and 65 to the precision published (`tests/test_model.py`).

## 3. Statutory reserves

*Code: `lamodel/reserves.py`*

Reserves are calculated per unit of benefit, per policy in force, on the **valuation** basis: the statutory table at the valuation interest rate $i_v$.

### 3.1 Net level premium (term)

$$
P=\frac{A^1_{x:\overline{n}|}}{\ddot a_{x:\overline{n}|}},\qquad
{}_tV^{NLP}=A^1_{x+t:\overline{n-t}|}-P\,\ddot a_{x+t:\overline{n-t}|}
$$

### 3.2 Commissioners Reserve Valuation Method (term)

CRVM replaces the level net premium with a lower first-year net premium $\alpha$ and a higher renewal net premium $\beta$. The difference allows for first-year expenses. The method has three steps.

**Step 1: first-year net premium and FPT renewal premium.**

$$
c_x = v\,q_x,\qquad
\beta^{FPT}=\frac{A^1_{x+1:\overline{n-1}|}}{\ddot a_{x+1:\overline{n-1}|}}
$$

**Step 2: cap the expense allowance.** The Standard Valuation Law limits the allowance to the 19-payment whole life renewal premium at age $x+1$, less $c_x$.

$$
E=\min\!\Big(\beta^{FPT}-c_x,\ \ \tfrac{A_{x+1}}{\ddot a_{x+1:\overline{19}|}}-c_x\Big)
$$

**Step 3: net premiums and reserve.**

$$
\beta=P+\frac{E}{\ddot a_{x:\overline{n}|}},\qquad \alpha=\beta-E,\qquad
{}_tV^{CRVM}=A^1_{x+t:\overline{n-t}|}-\beta\,\ddot a_{x+t:\overline{n-t}|}\quad(t\ge1)
$$

For level term the cap does not bind. CRVM therefore equals **full preliminary term**: $\alpha=c_x$ and ${}_1V=0$.

**Deficiency reserve.** If the gross premium $G<\beta$, the model adds $(\beta-G)\,\ddot a_{x+t:\overline{n-t}|}$.

**Recursion check (tested).** Each year,

$$
({}_tV+P_t)(1+i_v)=q_{x+t}+p_{x+t}\,{}_{t+1}V
$$

where $P_t=\alpha$ in the first year and $\beta$ thereafter.

### 3.3 CARVM (deferred income annuity)

The contract has no cash surrender value, so the only benefit stream the holder can elect is the contractual one. CARVM, which takes the greatest PV of future guaranteed benefits over all elective streams, is therefore

$$
{}_tV^{CARVM}=I\cdot{}_{(m-t)^+|}\ddot a_{x+t}\;+\;\mathbb 1_{t<m}\cdot SP\cdot A^1_{x+t:\overline{m-t}|}
$$

where:
- $I$ is the annual income,
- $m$ is the deferral period,
- $SP$ is the return-of-premium death benefit,
- the basis is the 2012 IAR at $i_v$.

**Recursion check (tested).**

$$
({}_tV-I_t)(1+i_v)=q\cdot DB_t+p\cdot{}_{t+1}V
$$

## 4. Cash-flow projection and profit test

*Code: `lamodel/term_life.py`, `lamodel/deferred_annuity.py`*

The projection is annual, for one policy issued at time 0, with expected decrements.

**In-force and decrements**

$$
\text{IF}_{t+1}=\text{IF}_t\,(1-q_t)(1-w_t)
$$

Deaths occur first, then lapses ($w_t$). The term contract expires at year 20.

**Statutory gain**

$$
\text{Gain}_t=\underbrace{G_t-\text{Comm}_t-\text{PremTax}_t-\text{Exp}_t}_{\text{start of year}}
+\text{Inv}_t-\text{Claims}_t-\Delta\text{Res}_t
$$

with investment income on the opening balance

$$
\text{Inv}_t=i_e\,(\text{Res}_{t-1}+G_t-\text{Comm}_t-\text{PremTax}_t-\text{Exp}_t)
$$

**Required capital** is a simplified RBC-style factor:

| Product | Required capital |
|---|---|
| Term | $RC_t=0.10\%\cdot\text{NAR}_t+2\%\cdot\text{Res}_t$ |
| DIA | $RC_t=4\%\cdot\text{Res}_t$ |

**Distributable earnings**

$$
DE_0=-RC_0,\qquad
DE_t=(1-\tau)\,\text{Gain}_t+(1-\tau)\,i_e\,RC_{t-1}-(RC_t-RC_{t-1})
$$

where $\tau=21\%$ is the tax rate.

**Measures**

| Measure | Definition |
|---|---|
| IRR | The root of $\sum_t DE_t(1+r)^{-t}=0$. Returned only when $DE$ changes sign exactly once, which by Descartes' rule guarantees a unique root; otherwise reported as undefined. |
| PV of profit | $\text{PV}_h(DE)$ at the hurdle rate $h=10\%$ |
| Profit margin | $\text{PV}_h(DE)/\text{PV}_h(G)$ |
| Breakeven year | The first year with non-negative cumulative $DE$ |

## 5. Pricing

The gross premium $G$ (term) or annual income $I$ (DIA) is solved by bisection so that

$$
\text{PV}_h(DE)=0\iff\text{IRR}=h
$$

Rate cards repeat the solve across issue ages and sexes (term), and across issue ages and deferral periods (DIA).

## 6. Premium build-up

*Code: `lamodel/analysis.py`*

**Term.** Every line of the profit test is discounted at the hurdle rate, with each year's items at the end of the year. Because $\text{PV}_h(DE)=0$ at the solved price, the identity below holds exactly (tested):

$$
\text{PV}_h(G)=\text{PV}_h(\text{Claims}+\text{Comm}+\text{PremTax}+\text{Exp}+\text{Tax})
+\underbrace{\text{PV}_h(\Delta\text{Res}-\text{Inv})}_{\text{reserve funding}}
+\underbrace{RC_0+\text{PV}_h(\Delta RC-(1-\tau)i_eRC)}_{\text{cost of capital}}
+\text{PV}_h(DE)
$$

**DIA.** The annuity is decomposed on the pricing view instead. Benefits and expenses are discounted at the earned rate with their actual timing, and the remainder of the premium is the margin that must fund tax, the cost of statutory reserves and capital, and profit.

## 7. Deterministic sensitivities

*Code: `lamodel/sensitivity.py`*

Each shock is run twice:

1. **At the base price**, giving the change in IRR and PV of profit.
2. **Repriced**, giving the new premium or payout rate that restores the hurdle.

The model also produces reserve grids across valuation interest, mortality and improvement, and a margin curve across earned rates from 3% to 7%.

## 8. Interest-rate scenarios

*Code: `lamodel/stochastic.py`*

### 8.1 Short-rate model

The model uses a Vasicek short rate,

$$
dr=a(b-r)\,dt+\sigma\,dW
$$

simulated with its exact annual transition:

$$
r_{t+1}=b+(r_t-b)e^{-a}+\sigma\sqrt{\tfrac{1-e^{-2a}}{2a}}\,Z
$$

The defaults are $r_0=b=4\%$, $a=0.12$ and $\sigma=1.1\%$.

### 8.2 New-money yield

The new-money rate is the analytic Vasicek 10-year zero yield plus a credit spread. The spread is calibrated so that long-run new money equals today's book yield. The zero-coupon bond price is $P(\tau)=A(\tau)e^{-B(\tau)r}$, where

$$
B(\tau)=\frac{1-e^{-a\tau}}{a},\qquad
\ln A(\tau)=\Big(b-\tfrac{\sigma^2}{2a^2}\Big)(B-\tau)-\tfrac{\sigma^2B^2}{4a}
$$

### 8.3 Portfolio earned rate

The portfolio earned rate rolls toward new money as 10% of assets are reinvested each year:

$$
y_{t+1}=0.9\,y_t+0.1\,\text{NM}_{t+1}
$$

This is a simple representation of reinvestment risk. Run on level paths at the base earned rate, the stochastic profit test reproduces the deterministic result exactly (tested).

## 9. Stochastic reserve

*Code: `lamodel/stochastic.py` (`gpvad`), `lamodel/analysis.py`*

Fix a valuation date $\tau$. Take the net liability outflows per policy in force on a prudent-estimate basis:

| Product | Prudent-estimate margins |
|---|---|
| Term | mortality +10%, lapses at 75% |
| DIA | mortality −10% |

Define, for each scenario $s$, the path discount factors $D^s_k=\prod_{j\le k}(1+y^s_j)^{-1}$. The **greatest present value of accumulated deficiencies** is

$$
\text{GPVAD}^s=\max\!\Big(0,\ \max_k\sum_{j\le k}\big(\text{Out}^{BOY}_j D^s_{j-1}+\text{Out}^{EOY}_j D^s_j\big)\Big)
$$

This is the smallest starting asset amount that keeps the asset balance non-negative at every future date. The model reports:

| Reserve | Definition |
|---|---|
| Stochastic reserve | $SR=\text{CTE}_{70}(\text{GPVAD})$ |
| Deterministic reserve | GPVAD on a level path at the base earned rate |
| Tail measure | $\text{CTE}_{98}$ (DIA) |

**Cross-check (tested).** With valuation mortality, no expenses and a level path at $i_v$, GPVAD for the DIA equals the CARVM reserve exactly.

These are the mechanics of the VM-20 and VM-22 stochastic reserve. The regulatory requirements (prescribed generators, asset modelling, tax, margins, exclusion tests) are not implemented.

## 10. Stochastic mortality

*Code: `lamodel/stochastic.py` (`simulate_term_block`)*

For a block of $N$ identical term policies, each simulation draws two sources of risk:

**Parameter risk.** A level multiplier $L\sim\text{Lognormal}$ with mean 1 and standard deviation 10%, drawn once per simulation. This represents the true mortality level differing from the pricing assumption.

**Process risk.** Binomial deaths and lapses each year:

$$
D_t\sim\text{Bin}(\text{IF}_t,\ L\,q_t),\qquad W_t\sim\text{Bin}(\text{IF}_t-D_t,\ w_t)
$$

The full statutory profit test is then run, vectorized over simulations, and the PV of block distributable earnings is summarized by its standard deviation, 99.5% VaR and CTE98.

As tested:
- the simulated mean matches the expected-value projection;
- the process-risk coefficient of variation scales with $1/\sqrt N$;
- parameter risk does not diversify.

## 11. Simplifications

* **Valuation framework.** Formula reserves follow the Standard Valuation Law (NLP, CRVM, CARVM). VM-20 (NPR/DR/SR), VM-22 and Regulation XXX are not implemented. Valuation interest rates are inputs.
* **Timing.** Annual and curtate: no monthly premiums or mid-year deaths, and terminal rather than mean reserves.
* **Tax.** Tax is charged at 21% on statutory gain. There are no tax reserves, DAC tax or loss carryforward limits.
* **Assets.** There is no asset model beyond the portfolio-yield roll; no ALM, hedging or credit defaults.
* **Pricing assumptions.** The term product has no preferred classes or post-level-term behaviour. The DIA has no selection factor. Expenses and capital factors are illustrative.
