# Interview Notes

A study guide for explaining this system. Each section gives the concept in plain terms,
then the question an interviewer is likely to ask and a model answer tied to the code.

---

## 1. The one-minute pitch

> "It's a continuation strategy with an ML filter. A simple rule finds impulse-then-pause
> setups and picks the direction. A LightGBM classifier estimates the probability that each
> setup reaches a 2 ATR target before a 1 ATR stop, and I only trade the high-probability
> ones. Labels come from the triple-barrier method, validation is purged walk-forward, the
> backtest charges fees and slippage on every fill, and the live engine imports the same
> feature, labeling and risk code so research and production can't drift apart."

Then be ready to say what you would **not** claim: the committed parameters are
placeholders and the repo shows a sound process, not a proven edge.

---

## 2. Core concepts

### Look-ahead bias
Using information at time *t* that only became available later. Examples: trading on a
bar's close when you could only have known its open; normalising with the mean of the whole
dataset; using a candle that has not finished forming.

**Q: How do you know your features have no look-ahead?**
Three layers. Structurally, I only use backward-looking operators — `rolling`, `ewm`,
positive `shift` — and no statistics fitted on the full series. By convention, a row is
known at its bar's close and everything acts on the next bar. And it is tested: I recompute
features on a truncated history and on a history where every future price is multiplied by
ten, and assert the past rows are identical. If anything peeked forward, it would change.

### ATR (Average True Range)
The average size of a bar, counting gaps. It is the unit of "how much does this move".

**Q: Why are your stops, targets and features in ATR units?**
A fixed 1% stop is tight when the market is wild and loose when it is quiet. ATR units
adapt to volatility, make setups on different assets comparable so I can pool them into one
model, and make position sizing volatility-aware automatically.

### Meta-labeling
A primary rule decides direction and timing; a secondary ML model decides whether to act.

**Q: Why not have the model predict direction on every bar?**
That asks the model to predict the market, which is nearly pure noise. Meta-labeling asks a
much narrower question — "is this specific setup a good one?" — gives a healthier class
balance, and keeps the strategy explainable because the model can only veto trades.

### Triple-barrier method
Label a trade by which is hit first: a profit target, a stop-loss, or a time limit.

**Q: Why not label by the return N bars ahead?**
Because that ignores how the trade is managed. A position with a stop can be knocked out
before bar N even if price ends higher; the fixed-horizon label would call a losing trade a
win. Triple-barrier labels are path-dependent and match the live exit logic.

**Q: What happens when one bar hits both barriers?**
OHLC data does not show which came first, so I assume the stop. A backtest should err
against itself.

**Q: What was the entry-bar problem?**
My first version also applied "stop first" on the entry bar. But when a long fills intrabar
on a move up through the trigger, the bar's low usually printed *before* the fill. Counting
it produced a large negative bias: on a random walk the average trade was about −0.35R when
it should be near zero, and almost half of all stop-outs were on the entry bar. I changed
the rule so an entry-bar stop counts only if the bar closes through the stop, unless the
fill was at the open. I found it by checking the labeler against a case where I knew the
right answer.

### Sample uniqueness
Consecutive bars often flag the same setup, so their labels overlap in time and are not
independent.

**Q: How do you handle overlapping labels?**
Each label is weighted by its average uniqueness: for each bar in its lifespan take one
over the number of labels alive on that bar, then average. Ten near-duplicate setups then
count for roughly one. The purge in cross-validation handles the same overlap across the
train/test boundary.

### Walk-forward validation, purging, embargo
Train on the past, test on the next block, roll forward.

**Q: Why not k-fold cross-validation?**
K-fold trains on the future and tests on the past, and because neighbouring samples
overlap, a shuffled test row has near-copies in the training set. Both inflate results.

**Q: What exactly does purging remove?**
A label is known at `t1`, when a barrier is hit, not at the setup time. A training sample
that starts before the test block but resolves inside it was built from test-period prices.
I keep a sample for training only if `t1 + embargo < test_start`.

**Q: What is the embargo for?**
An extra gap after the purge. Features are rolling windows and market states are serially
correlated, so samples either side of the boundary are very similar. To be precise about
it: in a walk-forward scheme the training data is always earlier than the test data, so the
purge does most of the work and the embargo is a conservative buffer. It matters more in
k-fold variants where training data also follows the test block.

### The placebo test
**Q: How would you detect a leak you had not thought of?**
Run the entire pipeline on a random walk. There is nothing to predict, so a leak-free
pipeline must score about AUC 0.5 and must not beat the no-skill log-loss baseline. If it
shows a strong edge on noise, the edge is coming from the future. That test is in CI.

A subtlety worth mentioning: even on noise the label is slightly predictable from
*volatility* features, because how often the time barrier is hit depends on volatility
relative to ATR. That is predictability of the label, not of direction, and it does not
translate into profit.

### Calibration
A model is calibrated if, among trades it scores at 0.6, about 60% win.

**Q: Why did you not re-balance the classes?**
I trade on a probability threshold. Re-weighting or resampling inflates predicted
probabilities, so the threshold would stop meaning what it says. I keep the natural class
balance and print a calibration table after training to check predicted against realised.

### Precision, recall, log-loss
- **Precision** — of the trades taken, how many won. This drives PnL.
- **Recall** — of all winners, how many were taken. This is opportunity cost.
- **Log-loss** — scores the probabilities themselves, compared with always predicting the
  base rate. If the model cannot beat that baseline, it has learned nothing.

---

## 3. Backtesting

**Q: What costs do you model?**
Taker fee plus adverse slippage on the entry, the stop exit and the time exit, because
those cross the spread. Maker fee and no slippage on the profit target, because it is a
resting limit order.

**Q: How much do costs matter?**
A round trip costs roughly 0.25–0.3% of notional. If the stop is 0.5% away, that is more
than half an R per trade. For a short-horizon strategy, costs decide viability more than
the model does.

**Q: What are the weaknesses of your backtest?**
Equity is marked at trade close, so drawdown inside a trade is understated. Slippage is a
flat number of basis points, while real breakout slippage scales with size and speed. There
is no funding or partial-fill modelling. And picking the probability threshold from the
same results you report is overfitting.

**Q: Explain your metrics.**
- **Sharpe** — mean return over standard deviation, annualised. Reward per unit of total
  volatility.
- **Sortino** — same, but only downside deviation in the denominator; upside is not risk.
- **Maximum drawdown** — worst peak-to-trough fall in equity.
- **Profit factor** — gross profit over gross loss.
- **Win rate** — meaningless alone; with a 2:1 payoff, 35% is profitable.
- **Expectancy in R** — average net result per trade in units of initial risk.

Daily returns include flat days. Dropping days without trades overstates Sharpe.

---

## 4. Risk management

**Q: How do you size a position?**
Fixed-fractional risk: `quantity = equity × risk% ÷ |entry − stop|`. With 10,000 equity,
1% risk and a stop 50 away, that is 2 units, and a stop-out loses 100. Because the stop is
one ATR, size shrinks when volatility rises.

**Q: Why 1%?**
Losses compound. Ten straight losses at 1% is a 9.6% drawdown; at 5% it is 40%, which needs
a 67% gain to recover. A strategy winning about a third of the time will have long losing
streaks.

**Q: Does the sizing formula ever fail?**
Yes: it ignores leverage. If ATR is 0.5% of price, 1% risk implies a position worth twice
the account. A notional cap shrinks the order, which means realised risk is below 1% — the
cap can only reduce risk.

**Q: Describe the kill-switch.**
It trips when equity falls 3% from the day's high-water mark, or after a run of consecutive
API errors. It is a latch: recovering later the same day does not re-enable trading. The
thresholds are configurable but clamped by ceilings in code, so a config typo or a tempted
operator can make it stricter but not looser. API-error halts need a manual restart.

**Q: Why halt on API errors?**
If the connection is failing I cannot trust my view of orders and positions. The safe
response to not knowing my state is to stop adding risk.

---

## 5. Execution engineering

**Q: Why stop-limit entries instead of market orders?**
A breakout strategy buys when everyone else is buying, so a market order pays whatever a
thin book asks. The stop means I only enter once price confirms the breakout; the limit
caps the worst price I accept. A missed trade costs nothing; a bad fill has negative
expectancy from the first tick.

**Q: What if your bot crashes while holding a position?**
The stop-loss is an order resting on the exchange, placed as soon as the entry fills, so
the position is protected without the bot. If that protective order cannot be placed, the
engine closes the position at market immediately.

**Q: A request times out after you send an order. What do you do?**
I do not know whether the exchange received it, and blindly retrying could double the
position. Every order gets a unique client order id before the first attempt. On retry I
first look for an open order with that id and adopt it. That makes placement idempotent.

**Q: Which errors do you retry?**
Transient ones — timeouts, network errors, rate-limit responses — with exponential backoff
and jitter. Permanent ones — insufficient funds, invalid order, authentication — fail
immediately, because retrying cannot help.

**Q: How do you avoid train/serve skew?**
The engine imports the research feature, setup and barrier functions; there is no second
implementation. The model is saved with a signature of the feature and label settings and
refuses to load if the config has changed. And a test checks that features computed on the
live rolling buffer match those computed on the full history.

**Q: Why asyncio?**
The bot mostly waits — for bars and for order acknowledgements. One thread can watch
several symbols and poll order status without blocking on a slow request. A single lock
serialises state changes, because correctness matters more than parallelism when a bar
arrives once an hour.

**Q: How is the engine tested without an exchange?**
The engine depends on two small interfaces, a market feed and a broker. In tests a replay
feed plays back bars and a paper broker simulates fills with the same rules as the labeler.
The engine cannot tell the difference, so the code path that trades real money is the one
the tests run.

---

## 6. Repository and security

**Q: How do you keep secrets and your edge out of a public repo?**
Keys are read from environment variables only and are excluded from `repr`. Tuned
parameters live in a gitignored `config/settings.yaml`; only a placeholder template is
committed. Model weights, data caches and research output are gitignored and also excluded
from the Docker build context, so they cannot end up in an image layer.

**Q: What does CI do?**
Lint, format check and the full test suite on every push and pull request, on two Python
versions, with read-only permissions and no secrets, using synthetic data.

---

## 7. What I would do next

- Use lower-timeframe bars to resolve the entry bar and same-bar barrier ambiguity exactly.
- Mark the backtest to market bar by bar for a true intra-trade drawdown.
- Model slippage as a function of order size relative to volume.
- Persist engine state and reconcile against the exchange on restart.
- Use the venue's native OCO orders for spot take-profit.
- Select the probability threshold on a dedicated validation period, and report a
  deflated Sharpe ratio to account for the number of configurations tried.
- Add probability calibration (isotonic or Platt) inside each training fold.
