class PortfolioRiskManager:
    def __init__(self, initial_portfolio=100_000, max_monthly_drawdown_pct=5):
        self.initial_portfolio = initial_portfolio
        self.portfolio_value = initial_portfolio
        self.max_monthly_drawdown_pct = max_monthly_drawdown_pct
        self.monthly_loss = 0

    def can_take_trade(self, position_pct, stop_loss_pct):
        """
        Checks if the proposed trade fits within the remaining risk budget.
        """
        position_size = self.portfolio_value * (position_pct / 100)
        potential_loss = position_size * (stop_loss_pct / 100)

        # Remaining risk allowed this month
        max_loss = self.portfolio_value * (self.max_monthly_drawdown_pct / 100)
        remaining_loss_budget = max_loss - self.monthly_loss

        return potential_loss <= remaining_loss_budget

    def take_trade(self, position_pct, stop_loss_pct, hit_stop=False):
        """
        Simulates taking a trade. If hit_stop=True, applies the loss.
        """
        position_size = self.portfolio_value * (position_pct / 100)
        loss = position_size * (stop_loss_pct / 100)

        if hit_stop:
            self.portfolio_value -= loss
            self.monthly_loss += loss

        return {
            "portfolio_value": round(self.portfolio_value, 2),
            "monthly_loss": round(self.monthly_loss, 2),
            "remaining_risk_budget": round(
                self.portfolio_value * (self.max_monthly_drawdown_pct / 100) - self.monthly_loss, 2
            )
        }

    def reset_month(self):
        """Reset monthly drawdown tracker (e.g. at start of new month)"""
        self.monthly_loss = 0
