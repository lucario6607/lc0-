#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <limits> // For std::numeric_limits

#include "chess/board.h"
#include "search/search.h" // For SearchBase
#include "utils/optionsdict.h" // For OptionsDict
#include "search/classic/node.h" // For classic::Node
#include "neural/network.h"    // For NodeLimits, EvalResult, GoParams
#include "neural/memcache.h"     // For NNCache
#include "search/classic/stoppers/stoppers.h" // For StoppersHints

namespace lczero {

// Helper struct for Gumbel distribution
struct GumbelDistribution {
    GumbelDistribution(double mu, double beta) : mu_(mu), beta_(beta) {}

    double Sample() {
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        double u = unif(rng_);
        return mu_ - beta_ * std::log(-std::log(u));
    }

private:
    double mu_;
    double beta_;
    std::mt19937 rng_{std::random_device{}()}; // Consider seeding properly
};

struct GumbelSearchParams {
    int m_search = 16;       // Number of top actions to consider (num_top_actions in issue)
    double c_scale = 1.0;    // Scale factor for value estimate
    int k_visit = 8;         // Number of visits to trigger backup
    int seed = -1;           // Seed for randomness, -1 for random device
    bool use_completed_q = true; // From issue's Gumbel code
    // Add other parameters as needed, e.g., FPU reduction
};

struct GumbelNodeData {
    double q_value = 0.0;
    int visits = 0;
    std::vector<Move> legal_moves;
    std::vector<double> policy_probs;
    std::vector<double> gumbel_values; // Stores Gumbel samples
    std::vector<int> visit_counts;    // N(s,a)
    std::vector<double> q_sa_values;   // Q(s,a)
    bool is_expanded = false;

    GumbelNodeData() = default; // Ensure default constructor

    // Constructor to initialize with capacity for m_search
    explicit GumbelNodeData(size_t m_search_capacity)
        : gumbel_values(m_search_capacity),
          visit_counts(m_search_capacity, 0),
          q_sa_values(m_search_capacity, 0.0) {}


    void Initialize(const std::vector<Move>& moves, const std::vector<double>& probs, double parent_q, const GumbelSearchParams& params) {
        legal_moves = moves;
        policy_probs = probs;
        gumbel_values.assign(moves.size(), 0.0);
        visit_counts.assign(moves.size(), 0);
        q_sa_values.assign(moves.size(), 0.0); // Initialize Q(s,a)

        GumbelDistribution gumbel_dist(0.0, 1.0); // mu=0, beta=1 for standard Gumbel
        for (size_t i = 0; i < moves.size(); ++i) {
            gumbel_values[i] = std::log(policy_probs[i] + 1e-9) + gumbel_dist.Sample();
        }
        is_expanded = true;

        // Apply Q_init transformation if needed (e.g., parent_q or value head)
        // For simplicity, let's assume Q_init is based on parent_q here.
        // A more sophisticated approach might use the value head from NN eval.
        for (size_t i = 0; i < moves.size(); ++i) {
            q_sa_values[i] = parent_q; // Initialize Q(s,a) with parent's Q value
        }
    }
};


class GumbelSearch : public SearchBase {
public:
    // OptionIds defined in engine.cc - need to access them or redefine them here.
    // For now, assume they are accessible or define placeholders if build fails.
    // This is a common issue when splitting definitions and declarations.
    // Let's assume for now they are linked correctly or use string literals.
    // For a cleaner solution, OptionIds could be in a shared header.

    GumbelSearch(UciResponder* responder, const OptionsDict* options)
        : SearchBase(responder) {
        params_.m_search = options->Get<int>("GumbelTopActions");
        // GumbelCScale is int, needs conversion to double c_scale
        params_.c_scale = static_cast<double>(options->Get<int>("GumbelCScale")) / 100.0;
        params_.k_visit = options->Get<int>("GumbelKVisit");
        params_.seed = options->Get<int>("GumbelSeed");

        if (params_.seed != -1) {
            rng_.seed(params_.seed);
        } else {
            rng_.seed(std::random_device{}());
        }
    }

    ~GumbelSearch() override = default;

    // Interface from SearchBase
    void SetPosition(const GameState& state) override {
        current_pos_ = state;
        root_node_ = std::make_unique<classic::Node>(nullptr, Move{}, current_pos_.startpos.GetBoard().IsWhiteToPlay(), 0.0f); // TODO: prior
        node_data_.clear();
        // Ensure root node data is created if not already
        if (node_data_.find(root_node_.get()) == node_data_.end()) {
            node_data_[root_node_.get()] = GumbelNodeData(params_.m_search); // Initialize with capacity
        }
    }

    void StartSearch(const GoParams& params) override {
        go_params_ = params;
        // Reset relevant search state if necessary
        // e.g., clear node_data_ or reset counters
        // Start the search loop/threads
        // For now, let's simulate a few playouts
        for (int i = 0; i < 1000; ++i) { // Example: 1000 playouts
            RunSinglePlayout(root_node_.get());
        }
    }
    void StopSearch() override { /* TODO: Implement stopping mechanism */ }
    void AbortSearch() override { /* TODO: Implement abort mechanism */ }
    void WaitSearch() override { /* TODO: Wait for search threads to finish */ }
    void NewGame() override {
        node_data_.clear();
        root_node_.reset(); // Or re-initialize as needed
    }
    void SetBackend(Backend* backend) override { nn_backend_ = backend; }
    void SetSyzygyTablebase(SyzygyTablebase* /*syzygy_tb*/) override { /* TODO: If Gumbel uses TB */ }
    void SetMaxNodes(NodeCount /*max_nodes*/) override { /* TODO: If applicable */ }
    void SetMaxTime(MilliSeconds /*max_time_ms*/) override { /* TODO: If applicable */ }
    NodeCount GetNodesVisited() const override { return nodes_visited_count_; }
    NodeCount GetNodesPerSecond() const override { return 0; /* TODO */ }
    Depth GetDepth() const override { return 0; /* TODO */ }
    std::vector<RootMoveInfo> GetRootMoveInfos() const override {
        std::vector<RootMoveInfo> infos;
        // Populate infos based on root_node_ and node_data_
        // This is a simplified example
        if (root_node_ && node_data_.count(root_node_.get())) {
            const auto& data = node_data_.at(root_node_.get());
            for (size_t i = 0; i < data.legal_moves.size() && i < (size_t)params_.m_search; ++i) {
                RootMoveInfo rmi;
                rmi.move = data.legal_moves[i];
                rmi.q = data.q_sa_values[i];
                rmi.n = data.visit_counts[i];
                // rmi.pv = ... // Would require more sophisticated PV tracking
                infos.push_back(rmi);
            }
            std::sort(infos.begin(), infos.end(), [](const RootMoveInfo& a, const RootMoveInfo& b){
                return a.n > b.n; // Example: sort by visits
            });
        }
        return infos;
    }
    Move GetBestMove() const override {
        auto infos = GetRootMoveInfos();
        if (!infos.empty()) {
            return infos[0].move;
        }
        return Move::NullMove(); // Should not happen in a proper search
    }

    void RunSimulationsFromNodes(const std::vector<classic::Node*>& start_nodes, int sims_per_node) {
        if (start_nodes.empty() || sims_per_node <= 0) return;

        // Assuming limits_, root_node_, and hints_ are accessible members of GumbelSearch
        // and are appropriately initialized (e.g., limits_ and hints_ from GoParams).
        for (int i = 0; i < sims_per_node; ++i) {
            for (classic::Node* sn : start_nodes) {
                // Ensure limits_ and hints_ are checked correctly.
                // The original Gumbel code had:
                // if (limits_.StopsNow(root_node_->GetN(), hints_)) return;
                // This needs root_node_ to be the main root of the search for GetN() to make sense
                // in the context of global search limits. If root_node_ in GumbelSearch can change
                // or represents a sub-search, this check might need care.
                // For now, assume root_node_ is the global root and GetN() is its total visits.
                if (limits_.StopsNow(root_node_ ? root_node_->GetN() : 0, hints_)) return;
                RunSinglePlayout(sn);
            }
        }

        // Conceptual parallel implementation using a hypothetical thread_pool_:
        // if (!thread_pool_) { /* fallback to sequential as above */ }
        // else {
        //     for (int i = 0; i < sims_per_node; ++i) {
        //         for (classic::Node* sn : start_nodes) {
        //             if (limits_.StopsNow(root_node_ ? root_node_->GetN() : 0, hints_)) break;
        //             thread_pool_->AddTask([this, sn]() {
        //                 this->RunSinglePlayout(sn);
        //             });
        //         }
        //         if (limits_.StopsNow(root_node_ ? root_node_->GetN() : 0, hints_)) break;
        //     }
        //     thread_pool_->WaitIdle(); // Wait for all tasks in this batch to complete.
        // }
    }

private:
    void RunSinglePlayout(classic::Node* start_node) {
        classic::Node* current_node = start_node;
        ChessBoard board = current_pos_.startpos.GetBoard(); // TODO: Apply moves from GameState

        // SELECTION
        while (current_node->HasChildren() && !board.IsGameOver()) {
            GumbelNodeData& data = node_data_[current_node];
            if (!data.is_expanded) break; // Should be expanded if it has children usually

            Move best_move = SelectAction(current_node);
            if (best_move == Move::NullMove()) break; // No valid action

            // Find or create the child node
            classic::Node* next_node = nullptr;
            for (auto& child : current_node->GetChildren()) {
                if (child->GetMove() == best_move) {
                    next_node = child.get();
                    break;
                }
            }
            if (!next_node) {
                 // This case should ideally be handled by expansion or if SelectAction can pick unexpanded moves
                break;
            }
            board.ApplyMove(best_move);
            current_node = next_node;
        }

        // EXPANSION (if not already expanded or if it's a terminal/unexpanded leaf)
        if (!node_data_[current_node].is_expanded && !board.IsGameOver()) {
            ExpandNode(current_node, board);
        }

        // SIMULATION (using NN eval as substitute for MCTS-style simulation)
        // The "simulation" in Gumbel context is more about the backup of the evaluated Q value.
        // For non-terminal nodes, the Q-value comes from NN. For terminal, from game result.
        double value_estimate;
        if (board.IsGameOver()) {
            value_estimate = board.GetResult(current_pos_.startpos.IsWhiteToPlay()); // TODO: Ensure correct perspective
        } else {
            // For Gumbel, we often use the Q(s,a) from the parent after taking action 'a'
            // or we might re-evaluate the state 's''.
            // Here, we'll assume the value is estimated when the node was created or will be updated in backup.
            // Let's use a placeholder for now, as the value comes from backup logic.
            // A proper NN call would be:
            // EvalResult eval = nn_backend_->Evaluate(board);
            // value_estimate = eval.value;
            // For simplicity in this stage, we might just use the Q from node_data if available,
            // or propagate from children during backup.
            // Let's assume a simple case where expansion provides initial Qs.
            // The Q-value for the backup will be the Q_sa of the action taken to reach this state,
            // or if it's a newly expanded node, its own value estimate.
            // This part is tricky with Gumbel as it's not a direct simulation.
            // We will use the parent's Q_sa(s, best_action_from_s) for the backup.
            // The value_estimate for backup is derived from the selected action's Q_sa or NN eval of new state.
            // For now, let's assume ExpandNode populates initial Q values or we get it from parent.
            // This will be refined.
            EvalResult eval;
            if (nn_backend_) {
                 std::vector<const ChessBoard*> boards_to_eval = {&board};
                 std::vector<EvalResult> evals;
                 nn_backend_->Evaluate(boards_to_eval, &evals, NodeCount{0}); // Assuming Evaluate takes a vector
                 if (!evals.empty()) {
                    value_estimate = evals[0].value;
                 } else {
                    value_estimate = 0.0; // Fallback
                 }
            } else {
                value_estimate = 0.0; // Fallback if no backend
            }
        }


        // BACKUP
        Backup(current_node, value_estimate, start_node); // Pass start_node for k_visit check
        nodes_visited_count_++;
    }

    Move SelectAction(classic::Node* node) {
        GumbelNodeData& data = node_data_[node];
        if (!data.is_expanded || data.legal_moves.empty()) {
            return Move::NullMove();
        }

        Move best_action = Move::NullMove();
        double max_value = -std::numeric_limits<double>::infinity();

        // Sort actions by Gumbel value + Q_completed / N_completed
        // This is the core of Gumbel AlphaZero selection.
        // Q_completed(s,a) = (N(s,a) * Q(s,a) + Q_backup) / (N(s,a) + 1)
        // For selection, we use Q(s,a) directly.
        // The paper suggests: argmax_a (g_a + c_scale * Q(s,a)) for selection during search
        // and for play, argmax_a (N(s,a)^(1/tau) * pi_a + g_a) or just N(s,a)
        // Let's use the search-time selection: g_a + c_scale * Q(s,a)

        std::vector<size_t> top_indices;
        // Create pairs of (value, index) to sort and find top m_search actions
        std::vector<std::pair<double, size_t>> action_values;
        for (size_t i = 0; i < data.legal_moves.size(); ++i) {
            action_values.push_back({data.gumbel_values[i] + params_.c_scale * data.q_sa_values[i], i});
        }

        // Sort to get top m_search actions based on g_a + c_scale * Q(s,a)
        std::sort(action_values.rbegin(), action_values.rend()); // Sort descending

        // Select from the top m_search actions
        // The paper's action selection for MCTS part (after initial M moves):
        // a_t = argmax_a ( Q(s,a) + U(s,a) ) where U is PUCT-like exploration.
        // Gumbel search selection is simpler: among the top M actions from initial sampling,
        // pick the one that maximizes Q_transformed(s,a) + sigma(N(s,a))
        // where Q_transformed is usually just Q(s,a) and sigma is an exploration bonus.
        // For now, let's pick the one with highest (g_a + c_scale * Q(s,a)) among top M,
        // or perhaps the one most visited, or with highest Q.
        // A common Gumbel strategy is to pick from the m_search top actions,
        // the one that maximizes Q(s,a) + exploration_bonus.
        // Let's try picking the action with highest (g_a + c_scale * Q_sa) from the top M.
        // This seems to be what "Sequential Halving" or similar methods would build upon.

        // For simplicity in this iteration, let's iterate through the top `m_search` actions
        // and select the one that maximizes Q(s,a) + exploration_term.
        // Or, even simpler for now: just pick the highest (g_a + c_scale * Q_sa) from the chosen set.
        // This part needs to align with the specific Gumbel variant being implemented.
        // Let's assume for now we pick the action with highest (gumbel_value + c_scale * Q_sa)
        // from the *subset* of top M actions.
        // The selection logic within the MCTS part of Gumbel can vary.
        // A simple approach: from the top M actions (by policy+Gumbel), pick the one maximizing Q + U.
        // For now, let's just pick the highest value one from the sorted list.
        if (!action_values.empty()) {
            best_action = data.legal_moves[action_values[0].second];
        }


        // If a child node resulting from best_action has N(s,a) >= k_visit,
        // then we need to re-evaluate Q(s,a) using backup from that child.
        // This is handled in the Backup phase.
        // The selection here just picks the current best.

        return best_action;
    }


    void ExpandNode(classic::Node* node, const ChessBoard& board) {
        if (node_data_[node].is_expanded) return;

        // Get legal moves and policy from NN
        EvalResult eval;
        if (nn_backend_) {
            std::vector<const ChessBoard*> boards_to_eval = {&board};
            std::vector<EvalResult> evals;
            // Pass NodeCount{0} as placeholder for nodes_evaluated_by_nn if not tracked here
            nn_backend_->Evaluate(boards_to_eval, &evals, NodeCount{0});
            if (!evals.empty()) {
                eval = evals[0];
            } else {
                // Handle error or provide default eval
                eval.value = 0.0f; // Default value
                // Policy might be uniform or based on heuristics if NN fails
            }
        } else {
            // Handle case where backend is not available (e.g., return uniform policy)
            eval.value = 0.0f;
        }


        std::vector<Move> legal_moves;
        std::vector<double> policy_probs;
        board.GetLegalMoves(&legal_moves);

        if (legal_moves.empty()) {
            node_data_[node].is_expanded = true; // Mark as expanded even if no moves
            return;
        }

        // If NN provided policy, map it to legal moves. Otherwise, uniform.
        if (!eval.policy.empty()) {
            policy_probs.reserve(legal_moves.size());
            for (Move m : legal_moves) {
                // Assuming eval.policy is a map or easily indexable by move
                // This needs to match how policy is structured in EvalResult
                // For now, let's assume it's a flat array corresponding to all possible moves
                // and we need to pick the ones for legal_moves.
                // This is a placeholder for actual policy extraction logic.
                // Example: policy_probs.push_back(eval.policy[move_to_idx(m)]);
                // If eval.policy is already filtered/mapped by NNWrapper, use directly.
                // For now, let's use uniform if policy logic is complex.
                // A common approach is that EvalResult.policy is a vector for *all* moves
                // and we need to pick the probabilities for the legal ones.
                // Let's find the policy for each legal move.
                // This requires that EvalResult.policy is structured in a way that we can look up policy by Move.
                // For simplicity, if eval.policy is not pre-filtered, we might need a mapping.
                // Let's assume eval.policy is a vector of (Move, prob) pairs or similar.
                // Or, if it's a flat list, we need to know the indexing scheme.
                // To keep it simple, let's use uniform distribution for now if policy mapping isn't straightforward.
                // TODO: Replace with actual policy mapping.
                 bool found_policy = false;
                 if (eval.policy.size() == legal_moves.size()) { // simplified assumption
                     for(size_t i=0; i<legal_moves.size(); ++i) {
                         // This assumes eval.policy is ordered same as legal_moves or is directly usable
                         // This part is highly dependent on how neural::Network formats policy.
                         // Let's assume for now that the network evaluation provides a policy vector
                         // that corresponds to the legal moves if they are passed to it, or a global policy.
                         // A robust implementation needs to map from NN output tensor to legal moves.
                         // policy_probs.push_back(eval.policy[i]); // Placeholder
                     }
                     // If the above placeholder is used, we need to ensure eval.policy is correctly populated.
                     // For now, let's fall back to uniform if policy is not easily mapped.
                     // found_policy = true;
                 }

                if (!found_policy) { // Fallback to uniform
                    policy_probs.assign(legal_moves.size(), 1.0 / legal_moves.size());
                } else {
                    // Renormalize if necessary
                    double sum_p = 0.0;
                    for (double p : policy_probs) sum_p += p;
                    if (sum_p > 1e-6) {
                        for (double& p : policy_probs) p /= sum_p;
                    } else { // All zero, fallback to uniform
                        policy_probs.assign(legal_moves.size(), 1.0 / legal_moves.size());
                    }
                }
            }
        } else { // No policy from NN, use uniform
            policy_probs.assign(legal_moves.size(), 1.0 / legal_moves.size());
        }


        // Initialize GumbelNodeData for the current node
        // Pass the parent's Q-value (or current node's NN eval if preferred for root/first expansion)
        // For non-root, Q_init is often based on parent's Q. For root, from NN eval.
        double q_init_value = (node->GetParent() == nullptr) ? eval.value : node_data_[node->GetParent()].q_value;
        node_data_[node].Initialize(legal_moves, policy_probs, q_init_value, params_);

        // Create child nodes
        for (size_t i = 0; i < legal_moves.size(); ++i) {
            Move move = legal_moves[i];
            // TODO: prior for child node? Usually policy_probs[i]
            auto child_node = std::make_unique<classic::Node>(node, move, !board.IsWhiteToPlay(), static_cast<float>(policy_probs[i]));
            node->AddChild(std::move(child_node));
            // Initialize GumbelNodeData for new children (important for m_search capacity)
            // Children data will be more fully populated when they are expanded.
            // Only ensure they exist in the map with correct capacity.
            classic::Node* child_ptr = node->GetChildren().back().get(); // Get pointer to newly added child
            if (node_data_.find(child_ptr) == node_data_.end()) {
                 node_data_[child_ptr] = GumbelNodeData(params_.m_search);
            }
             // Initialize Q(s,a) for the action leading to this child in parent's data.
             // This is Q_init(s,a) = V(s') if V(s') is from NN eval of child state.
             // Or, it could be simpler: Q_init(s,a) = V_parent_nn_eval.
             // The Initialize method of GumbelNodeData already sets initial Q(s,a) based on parent_q.
        }
    }

    void Backup(classic::Node* leaf_node, double value_estimate, classic::Node* start_node) {
        classic::Node* current_node = leaf_node;
        bool is_first_node_in_backup = true; // To handle value for the expanded/simulated node itself

        while (current_node != nullptr) {
            GumbelNodeData& data = node_data_[current_node];
            data.visits++; // N(s) updated

            if (current_node == start_node && current_node != root_node_.get()) {
                // This logic is for the k_visit rule. If we backed up to the node
                // from which a multi-step action sequence (a "plan") started,
                // and if the visit count for the *action* within that plan (at start_node)
                // reaches k_visit, then we need to re-evaluate Q(start_node, action_taken).
                // This is complex and might need careful state passing.
                // For now, let's simplify: update Q(s,a) at parent during backup.
            }


            classic::Node* parent = current_node->GetParent();
            if (parent) {
                GumbelNodeData& parent_data = node_data_[parent];
                Move move_to_current = current_node->GetMove();
                size_t action_idx = -1;
                for (size_t i = 0; i < parent_data.legal_moves.size(); ++i) {
                    if (parent_data.legal_moves[i] == move_to_current) {
                        action_idx = i;
                        break;
                    }
                }

                if (action_idx != (size_t)-1) {
                    parent_data.visit_counts[action_idx]++; // N(p,a)
                    // Q(p,a) = (N(p,a)-1 * Q(p,a) + V_current_node) / N(p,a)
                    // V_current_node is value_estimate if current_node is the leaf_node of playout,
                    // otherwise it's the Q_value of current_node (max_a' Q(current_node, a'))
                    double backup_value = is_first_node_in_backup ? value_estimate : data.q_value;
                    parent_data.q_sa_values[action_idx] =
                        ((parent_data.visit_counts[action_idx] - 1) * parent_data.q_sa_values[action_idx] + backup_value) /
                        parent_data.visit_counts[action_idx];

                    // Check k_visit condition for re-evaluation (more advanced)
                    // If parent_data.visit_counts[action_idx] == params_.k_visit:
                    //    Trigger re-evaluation of Q(parent, move_to_current)
                    //    This often means doing a 1-ply search from `current_node` (the state s')
                    //    and using its value (max_a' Q(s',a')) as the new Q(parent, move_to_current).
                    //    This is part of the "improved backup" in Gumbel-MuZero.
                    //    For now, the standard backup is implemented.
                }
            }

            // Update Q(s) for the current node based on its children's Q(s,a) values.
            // Q(s) = max_a Q(s,a) for the current node `data`.
            // This is used if Gumbel selection needs Q(s) or for passing up the tree.
            if (!data.q_sa_values.empty()) {
                double max_q_sa = -std::numeric_limits<double>::infinity();
                for (double q_val : data.q_sa_values) {
                    if (q_val > max_q_sa) {
                        max_q_sa = q_val;
                    }
                }
                data.q_value = max_q_sa; // Update node's overall Q value
            } else if (is_first_node_in_backup) {
                // If it's a leaf and has no actions (e.g. terminal), its Q is just the value_estimate
                data.q_value = value_estimate;
            }


            current_node = parent;
            is_first_node_in_backup = false;
        }
    }


private:
    GameState current_pos_;
    std::unique_ptr<classic::Node> root_node_;
    GoParams go_params_;
    GumbelSearchParams params_; // Renamed from gumbel_params_ in issue
    Backend* nn_backend_ = nullptr; // Not owned
    std::mt19937 rng_;
    GumbelDistribution gumbel_dist_; // Used for root children sampling

    PositionHistory current_pos_history_; // To store current game history
    GameState current_pos_; // Keep this if it's used by existing methods like RunSinglePlayout's board setup

    NodeLimits limits_; // For checking stop conditions
    StoppersHints* hints_{nullptr}; // For checking stop conditions

    std::atomic<bool> stop_requested_{false};
    std::mutex node_data_mutex_; // To protect node_data_ if accessed by multiple threads

    // Store Gumbel-specific data per node
    std::unordered_map<const classic::Node*, GumbelNodeData> node_data_;
    NodeCount nodes_visited_count_ = 0;


    // TODO: Add other necessary members like NN cache interface, stop conditions, thread_pool_, etc.
    // Placeholder for DoGumbelSearch and its helpers, to be defined in .cc
    void DoGumbelSearch();
    EvalResult ExpandAndGetEvaluation(classic::Node* node);
    // ExpandNode is called by ExpandAndGetEvaluation, its signature in issue is (Node*, MoveList, PolicyType)
    // PolicyType is std::vector<float> in issue.
    void ExpandNode(classic::Node* node, const MoveList& moves, const std::vector<float>& priors);
    classic::Node* SelectChild(classic::Node* node);
    void UpdateCompletedQValues(classic::Node* parent, const std::vector<classic::Node*>& active_candidates);
    // RunSinglePlayout is already in .h from previous step
    // BackupValue signature from issue: (const std::vector<Node*>& path, float value)
    void BackupValue(const std::vector<classic::Node*>& path, float value);


};

} // namespace lczero
