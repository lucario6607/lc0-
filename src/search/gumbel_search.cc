#include "search/gumbel_search.h"

#include <string_view>
#include <memory> // For std::make_unique
#include <atomic> // For std::atomic
#include <mutex>  // For std::mutex, std::lock_guard

#include "search/register.h"     // For SearchFactory, REGISTER_SEARCH
#include "utils/optionsdict.h"   // For OptionsDict
#include "utils/optionsparser.h" // For OptionsParser
#include "utils/exception.h"     // For Exception
#include "chess/position.h"      // For PositionHistory
#include "chess/gamestate.h"     // For GameState
#include "neural/network.h"      // For GoParams, NodeLimits
#include "search/classic/node.h" // For classic::Node (used by GumbelSearch members)


namespace lczero {

// Factory definition
class GumbelSearchFactory : public SearchFactory {
public:
    std::string_view GetName() const override { return "gumbel"; }

    void PopulateParams(OptionsParser* options_parser) const override {
        // Options are assumed to be populated globally by Engine::PopulateOptions for now.
        // If Gumbel-specific options needed further handling by the factory, it would go here.
        (void)options_parser; // Mark as unused
    }

    std::unique_ptr<SearchBase> CreateSearch(
        UciResponder* responder,
        const OptionsDict* options) const override {
        if (!options) {
            throw Exception("OptionsDict is null in GumbelSearchFactory::CreateSearch");
        }
        return std::make_unique<GumbelSearch>(responder, options);
    }
};

REGISTER_SEARCH(GumbelSearchFactory);


// GumbelSearch method implementations

GumbelSearch::GumbelSearch(UciResponder* responder, const OptionsDict* options_in)
    : SearchBase(responder),
      gumbel_dist_(0.0, 1.0) { // Initialize GumbelDistribution with mu=0, beta=1
    if (!options_in) {
        throw Exception("OptionsDict is null in GumbelSearch constructor");
    }

    // Initialize Gumbel-specific parameters
    params_.m_search = options_in->Get<int>("GumbelTopActions");
    params_.c_scale = static_cast<double>(options_in->Get<int>("GumbelCScale")) / 100.0;
    params_.k_visit = options_in->Get<int>("GumbelKVisit");
    params_.use_completed_q = options_in->Get<bool>("GumbelCompletedQ");

    if (auto seed_opt = options_in->TryGet<int>("GumbelSeed")) {
        params_.seed = *seed_opt;
        if (params_.seed == -1) {
            rng_.seed(std::random_device{}());
        } else {
            // mt19937 constructor/seed method usually takes unsigned int
            rng_.seed(static_cast<unsigned int>(params_.seed));
        }
    } else {
        params_.seed = -1; // Default if not found
        rng_.seed(std::random_device{}());
    }

    // root_node_ is initialized in SetPosition/NewGame
    // backend_ is set via SetBackend
    // limits_ and hints_ are set via StartSearch from GoParams
    // stop_requested_ is already initialized (atomic_bool)
    // node_data_mutex_ is already initialized
}

void GumbelSearch::SetBackend(Backend* backend_ptr) {
    backend_ = backend_ptr; // backend_ is an inherited member from SearchBase
}

void GumbelSearch::Clear() {
    std::lock_guard<std::mutex> lock(node_data_mutex_);
    node_data_.clear();
    if (root_node_) {
        root_node_.reset();
    }
    // Reset any other Gumbel-specific state if necessary
}

void GumbelSearch::NewGame() {
    SearchBase::NewGame(); // Call base class version if it exists and does something.
                           // SearchBase::NewGame() is virtual and might be overridden by specific searches.
    Clear();
    stop_requested_.store(false);
    // Reset any game-specific stats or counters here.
    // For example, nodes_visited_count_ from gumbel_search.h could be reset:
    nodes_visited_count_ = 0;
}

void GumbelSearch::SetPosition(const GameState& gs) {
    Clear();
    stop_requested_.store(false);

    // Store the game state. current_pos_ is already a member of GumbelSearch (added in gumbel_search.h)
    current_pos_ = gs;
    current_pos_history_ = PositionHistory(gs.startpos);
    for(const auto& move : gs.moves) {
        current_pos_history_.Append(move);
    }

    // Create a new root node for this position.
    // classic::Node constructor: Node(Node* p, Move m, bool is_white_to_move, float prior_probability)
    // Root has no parent or move that led to it. Pass prior of 0.0f or some default.
    const ChessBoard& board = current_pos_history_.Last().GetBoard();
    root_node_ = std::make_unique<classic::Node>(nullptr, Move::NullMove(), board.IsWhiteToPlay(), 0.0f);

    // Ensure GumbelNodeData for root is created.
    // Use emplace to construct in place if possible, or assignment.
    // The GumbelNodeData constructor taking capacity is explicit, so direct assignment is fine.
    node_data_[root_node_.get()] = GumbelNodeData(params_.m_search);
}

void GumbelSearch::StartSearch(const GoParams& params) {
    stop_requested_.store(false);
    limits_ = params.limits;
    hints_ = params.hints;

    // The main Gumbel search logic.
    // This might run in a separate thread in a full implementation.
    DoGumbelSearch();
}

void GumbelSearch::StopSearch() {
    stop_requested_.store(true);
}

void GumbelSearch::AbortSearch() {
    stop_requested_.store(true);
}

void GumbelSearch::WaitSearch() {
    // If StartSearch spawns a thread for DoGumbelSearch, this method should join that thread.
    // For now, assuming DoGumbelSearch is blocking or not yet implemented.
}

void GumbelSearch::StartClock() {
    // Timing is typically handled by the Engine or UciLoop.
    // This method might be a no-op if GumbelSearch doesn't manage its own detailed timing.
}

void GumbelSearch::DoGumbelSearch() {
    if (!root_node_) {
        throw Exception("Root node is null in DoGumbelSearch");
    }

    // Initial expansion of the root node if not already done
    if (!node_data_[root_node_.get()].is_expanded) {
        ExpandAndGetEvaluation(root_node_.get());
    }

    while (!stop_requested_.load() && !limits_.StopsNow(root_node_->GetN(), hints_)) {
        // Select K actions to run simulations on.
        // This part is high-level in the issue's GumbelSearch; it implies selecting promising children.
        // For MuZero-style Gumbel, this involves sampling M actions, then picking K for simulation.
        // Let's assume we get a set of candidate children nodes from the root.
        // The issue's `RunSearch` implies a loop that processes the root node's children.

        // The issue's `RunSearch` has:
        // 1. InitializeChildren(root_node_): Sample Gumbel values for root's children, sort them.
        //    This seems to be part of ExpandNode or initial setup of GumbelNodeData for root.
        //    My GumbelNodeData::Initialize does this.
        //
        // 2. Loop `while (!LimitsStop())`:
        //    `active_candidates = GetTopKCandidates(root_node_, params_.num_top_actions);`
        //    `RunSimulationsFromNodes(active_candidates, 1);`
        //    `UpdateCompletedQValues(root_node_, active_candidates);`
        //
        // `GetTopKCandidates` is not defined in the issue, but it implies selecting children.
        // Let's assume `root_node_->GetChildren()` provides the candidates for now,
        // and they are already sorted by Gumbel value + policy logit in `ExpandNode`.
        // Or, `SelectChild` could be used if we are picking one by one.
        // The `RunSimulationsFromNodes` in the issue's code takes a vector of nodes.

        // Simplified: Get all children of root as candidates.
        // A proper GetTopKCandidates would sort based on Gumbel value + Q_completed / N_completed.
        std::vector<classic::Node*> current_candidates;
        if (root_node_->HasChildren()) {
            for (const auto& child : root_node_->GetChildren()) {
                current_candidates.push_back(child.get());
            }
        } else {
            // If root has no children (e.g. terminal or not expanded fully), break or handle.
            // If it was just expanded, it should have children unless it's a terminal leaf.
            if (root_node_->IsTerminal()) break; // Nothing to search
            // Potentially expand again if it failed before and wasn't terminal
            if (!node_data_[root_node_.get()].is_expanded) {
                 ExpandAndGetEvaluation(root_node_.get()); // Try one more time
                 if (!root_node_->HasChildren() && !root_node_->IsTerminal()) break; // Still no children and not terminal
                 for (const auto& child : root_node_->GetChildren()) {
                    current_candidates.push_back(child.get());
                 }
            }
            if(current_candidates.empty()) break;
        }

        // Sort candidates? The issue's GetTopKCandidates implies sorting.
        // GumbelNodeData::Initialize already samples gumbel_values.
        // Let's assume they are implicitly sorted or selection logic in RunSimulations handles it.
        // For now, we pass all children. A more sophisticated approach would select top K.
        // The method RunSimulationsFromNodes (defined in .h) will run playouts.
        RunSimulationsFromNodes(current_candidates, 1); // 1 simulation per candidate in this iteration

        if (params_.use_completed_q) {
            UpdateCompletedQValues(root_node_.get(), current_candidates);
        }
    }
}

// ExpandAndGetEvaluation: Gets NN eval for a node, expands it by creating children.
EvalResult GumbelSearch::ExpandAndGetEvaluation(classic::Node* node) {
    if (!node) {
        // This case should ideally not be reached if logic is correct.
        // Return a default or throw, depending on how critical this is.
        return {}; // Default EvalResult
    }

    // Handle terminal nodes first
    // classic::Node::IsTerminal() checks if WDL is known.
    if (node->IsTerminal()) {
        // TODO: Convert node's terminal state (W, D, L from node's perspective)
        // to EvalResult {q, d, m, policy}.
        // q: value for current player at 'node'.
        // d: draw probability.
        // m: moves left (0 for terminal).
        // policy: empty.
        // This requires mapping node->GetW(), node->GetL(), node->GetD() to q, d.
        // node->GetQ(-1) is Q from parent's perspective. We need Q for current player at node.
        // For now, a simplified placeholder.
        return {node->GetQ(1), node->GetD(), 0.0f, {}}; // Q for current player, draw_rate, moves_left=0
    }

    if (!backend_) {
        throw Exception("GumbelSearch: Backend not set for NN evaluation.");
    }

    // Reconstruct PositionHistory for the current node to send to NN.
    // This is a critical and potentially complex part.
    // current_pos_history_ is for root_node_. For other nodes, we need to traverse.
    PositionHistory history_for_node;
    if (node == root_node_.get()) {
        history_for_node = current_pos_history_;
    } else {
        // Reconstruct path from root to node
        std::vector<classic::Node*> path;
        classic::Node* temp_n = node;
        while(temp_n && temp_n != root_node_.get()) {
            path.push_back(temp_n);
            temp_n = temp_n->GetParent();
        }
        if (!temp_n) throw Exception("Node not found in tree from root."); // Should not happen
        std::reverse(path.begin(), path.end());

        history_for_node = current_pos_history_;
        for(classic::Node* n_in_path : path) {
            history_for_node.Append(n_in_path->GetMove());
        }
    }

    const Position& current_board_pos = history_for_node.Last();
    MoveList legal_moves = current_board_pos.GetBoard().GenerateLegalMoves();

    if (legal_moves.empty()) { // Terminal state reached (checkmate or stalemate)
        GameResult result = current_board_pos.GetBoard().IsUnderCheck() ?
                            (current_board_pos.GetBoard().IsWhiteToPlay() ? GameResult::BLACK_WON : GameResult::WHITE_WON) :
                            GameResult::DRAW;
        node->MakeTerminal(result); // Mark the node as terminal
        // TODO: Convert result to EvalResult properly.
        if (result == GameResult::DRAW) return {0.0f, 1.0f, 0.0f, {}};
        // For win/loss, q is from perspective of player *at node*.
        // If current player at node lost, q = -1. If won, q = 1.
        // This needs to align with how node->MakeTerminal sets WDL.
        return { (result == GameResult::WHITE_WON && current_board_pos.GetBoard().IsWhiteToPlay()) ||
                 (result == GameResult::BLACK_WON && !current_board_pos.GetBoard().IsWhiteToPlay()) ? 1.0f : -1.0f,
                 0.0f, 0.0f, {}};
    }

    EvalResult eval;
    auto computation = backend_->CreateComputation();
    // Backend expects vector of Position objects for history.
    // PositionHistory::GetPositions() returns const std::vector<Position>&
    computation->AddInput({history_for_node.GetPositions(), legal_moves}, eval.AsPtr());
    computation->ComputeBlocking(); // This is simplified. Batching/async would be more complex.

    // The policy from NN (eval.p) is for legal moves.
    // ExpandNode in Gumbel context means creating children and initializing their Gumbel data.
    if (!node_data_[node].is_expanded) { // Check if Gumbel-specific expansion has occurred
                                     // classic::Node::IsExpanded() might mean something different.
        this->ExpandNode(node, legal_moves, eval.p); // eval.p is policy from NN
    }

    // NN output 'q' is from current player's perspective.
    // Gumbel often uses parent's perspective for Q values in some contexts.
    // For EvalResult.q, it should be from the perspective of the player to move at 'node'.
    // So, direct eval.q should be fine here.
    return eval;
}


// ExpandNode (Gumbel specific): Initializes children of 'node' with policy priors and Gumbel noise.
void GumbelSearch::ExpandNode(classic::Node* node, const MoveList& moves, const std::vector<float>& priors) {
    std::lock_guard<std::mutex> lock(node_data_mutex_); // Protect node_data_ and potentially node's children list

    if (node_data_[node].is_expanded || moves.empty()) {
        return;
    }

    std::vector<double> policy_double(priors.begin(), priors.end());
    double parent_q_for_init = node_data_[node].q_value; // Or from parent if available: node->GetParent() ? node_data_[node->GetParent()].q_value : 0.0;
                                                       // If node is root, its q_value might be from a previous eval or 0.

    node_data_[node].Initialize(moves, policy_double, parent_q_for_init, params_);

    // Create child classic::Node objects if they don't exist
    // This part is tricky as classic::Node manages its own children.
    // We need to ensure that GumbelNodeData is consistent with classic::Node children.
    if (!node->HasChildren() && !moves.empty()) { // Only add children if classic::Node doesn't have them
        for (size_t i = 0; i < moves.size(); ++i) {
            auto child_node = std::make_unique<classic::Node>(node, moves[i], !node->IsWhiteToMove(), priors[i]);
            classic::Node* child_ptr = child_node.get(); // Get pointer before moving
            node->AddChild(std::move(child_node));
            // Ensure GumbelNodeData for new children
            if (node_data_.find(child_ptr) == node_data_.end()) {
                 node_data_[child_ptr] = GumbelNodeData(params_.m_search);
            }
        }
    }
    node_data_[node].is_expanded = true; // Mark Gumbel-specific expansion
}

// SelectChild: Selects the best child of 'node' based on Gumbel criteria.
classic::Node* GumbelSearch::SelectChild(classic::Node* node) {
    std::lock_guard<std::mutex> lock(node_data_mutex_); // Protect node_data_

    if (!node_data_[node].is_expanded || !node->HasChildren()) {
        return nullptr;
    }

    classic::Node* best_child = nullptr;
    double max_value = -std::numeric_limits<double>::infinity();

    const GumbelNodeData& data = node_data_.at(node); // Use .at() for const access, relies on node being in map

    // Iterate over classic::Node children and match them with GumbelNodeData entries
    // This assumes children in classic::Node map directly to entries in GumbelNodeData's legal_moves/etc.
    // This requires careful synchronization if classic::Node children can change independently.
    // For now, assume GumbelNodeData.legal_moves matches the order of children or can be indexed.

    size_t child_idx = 0;
    for (const auto& child_node_ptr : node->GetChildren()) {
        if (child_idx >= data.gumbel_values.size()) break; // Should not happen if synced

        // The selection criteria from issue: Q(s,a) + GumbelValue(s,a) / N(s,a)
        // Or, if N(s,a) is 0, use Q_completed(s,a) + GumbelValue(s,a)
        // Q(s,a) is data.q_sa_values[child_idx]
        // GumbelValue(s,a) is data.gumbel_values[child_idx]
        // N(s,a) is data.visit_counts[child_idx]

        double q_val = data.q_sa_values[child_idx];
        double g_val = data.gumbel_values[child_idx]; // This is log P(a|s) + G_noise
                                                      // The issue's SelectChild uses `child.gumbel_action_value`
                                                      // which is `log P(a|s) + G_noise`.
                                                      // And `child.Q()` which is `Q(s,a)`.
                                                      // Selection is `argmax_a (Q(s,a) + G(s,a))`
                                                      // where G(s,a) is the Gumbel sample with policy.

        double current_value = q_val + g_val; // This matches typical Gumbel search selection.

        if (current_value > max_value) {
            max_value = current_value;
            best_child = child_node_ptr.get();
        }
        child_idx++;
    }
    return best_child;
}

// UpdateCompletedQValues: Updates Q-values based on simulations, part of Gumbel-MuZero.
void GumbelSearch::UpdateCompletedQValues(classic::Node* parent, const std::vector<classic::Node*>& active_candidates) {
    std::lock_guard<std::mutex> lock(node_data_mutex_); // Protect node_data_

    if (!node_data_[parent].is_expanded) return;

    GumbelNodeData& parent_g_data = node_data_[parent];

    for (classic::Node* child_node : active_candidates) {
        // Find index of child_node in parent_g_data.legal_moves
        Move child_move = child_node->GetMove();
        size_t action_idx = -1;
        for (size_t i = 0; i < parent_g_data.legal_moves.size(); ++i) {
            if (parent_g_data.legal_moves[i] == child_move) {
                action_idx = i;
                break;
            }
        }
        if (action_idx == (size_t)-1) continue; // Should not happen

        if (parent_g_data.visit_counts[action_idx] >= params_.k_visit) {
            // Re-estimate Q(parent, child_action) using the completed Q-value of the child state.
            // Q_completed(s') = max_a' Q(s',a') from child's GumbelNodeData.
            // This is node_data_[child_node].q_value (which is max_a Q(child, a'))
            double child_s_q_value = node_data_[child_node].q_value;

            // The value from child's perspective needs to be negated for parent.
            parent_g_data.q_sa_values[action_idx] = -child_s_q_value;
        }
    }
}


// BackupValue: Propagates value up the tree path.
void GumbelSearch::BackupValue(const std::vector<classic::Node*>& path, float value) {
    // (void)path; (void)value; // Mark as unused for now
    // TODO: Implement actual Q-value and visit count updates for nodes in path.
    // This requires understanding how classic::Node's Q (W/L), D, M values and N (visits)
    // should be updated from a single playout's terminal value in Gumbel context.
    // The classic MCTS uses node->FinalizeScoreUpdate(v, d, m, 1);
    // Gumbel may or may not use WDL directly, often just Q.
    // For now, this is a placeholder. The logic in RunSinglePlayout that calls this
    // will use the GumbelNodeData map for updates.
    // This BackupValue is more for MCTS-style direct node updates if needed.
    // The Gumbel paper's backup is implicit in updating N(s,a) and Q(s,a) in GumbelNodeData.
    // The provided code uses Backup(current_node, value_estimate, start_node) in RunSinglePlayout,
    // which updates GumbelNodeData. This BackupValue might be redundant or for a different purpose.
    // Let's assume the Backup method in gumbel_search.h (from previous version) does the job.
    // This specific BackupValue might not be needed if that one is used.
    // The one in .h is: void Backup(classic::Node* leaf_node, double value_estimate, classic::Node* start_node);
    // The RunSinglePlayout in the issue calls BackupValue(path, eval.v);
    // This implies a path-based update.

    float current_q = value;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        classic::Node* node = *it;
        if (!node) continue;

        std::lock_guard<std::mutex> lock(node_data_mutex_); // Protect node_data_
        GumbelNodeData& g_data = node_data_[node];

        // This is a simplified MCTS-like update. Gumbel's primary update is via Q(s,a) in parent.
        // For the node s itself:
        g_data.visits++; // N(s)
        // Q(s) = ( (N(s)-1)*Q(s) + backed_value_for_s ) / N(s)
        // The backed_value_for_s for node 's' would be 'current_q' if 's' is the leaf of this path segment.
        // Or, it's derived from its children's Q(s,a) if it's an internal node in the path.
        // This update is for the node's overall state-value Q(s), not Q(s,a).
        // g_data.q_value = ((g_data.visits - 1) * g_data.q_value + current_q) / g_data.visits;


        // If this node has a parent in the path, update Q(parent, action_to_node)
        classic::Node* parent = node->GetParent();
        if (parent && (it + 1 != path.rend()) && *(it + 1) == parent ) { // Check if parent is next in reversed path
            Move move_to_node = node->GetMove();
            GumbelNodeData& parent_g_data = node_data_[parent];
            size_t action_idx = -1;
            for(size_t i=0; i < parent_g_data.legal_moves.size(); ++i) {
                if (parent_g_data.legal_moves[i] == move_to_node) {
                    action_idx = i;
                    break;
                }
            }
            if (action_idx != (size_t)-1) {
                parent_g_data.visit_counts[action_idx]++; // N(p,a)
                parent_g_data.q_sa_values[action_idx] =
                    ((parent_g_data.visit_counts[action_idx] - 1) * parent_g_data.q_sa_values[action_idx] + current_q) /
                    parent_g_data.visit_counts[action_idx];
            }
        }
        current_q = -current_q; // Flip value for the parent.
    }
}

// RunSinglePlayout: Simulates one playout from start_node.
// This was already in gumbel_search.h's GumbelSearch class from a previous step.
// The logic needs to be moved here if it's not already, or reconciled.
// The version in .h was a direct copy of the issue's RunSinglePlayout.
// Let's ensure it's here and uses the class members correctly.
// (Assuming it was moved from .h to .cc or was a placeholder in .h)
// The version in .h already calls a method named Backup(...), not BackupValue(...).
// I will keep the existing Backup and RunSinglePlayout from the .h file (which were copied from issue).
// The BackupValue here is distinct and called by the issue's version of RunSinglePlayout.
// This might mean the issue's RunSinglePlayout is slightly different from the one I have.
// The one I have (copied into .h from an earlier step) is:
/*
    void RunSinglePlayout(classic::Node* start_node) {
        classic::Node* current_node = start_node;
        ChessBoard board = current_pos_.startpos.GetBoard(); // TODO: Apply moves from GameState

        // SELECTION
        while (current_node->HasChildren() && !board.IsGameOver()) { ... Move best_move = SelectAction(current_node); ... }
        // EXPANSION
        if (!node_data_[current_node].is_expanded && !board.IsGameOver()) { ExpandNode(current_node, board); }
        // SIMULATION (NN eval)
        // ... value_estimate = ...
        // BACKUP
        Backup(current_node, value_estimate, start_node); // This calls the 3-arg Backup
        nodes_visited_count_++;
    }
    void Backup(classic::Node* leaf_node, double value_estimate, classic::Node* start_node) { ... }
*/
// The issue's Gumbel code has a RunSinglePlayout that calls SelectChild, ExpandAndGetEvaluation, and BackupValue(path, value).
// This is a conflict. I will use the issue's version of RunSinglePlayout for this subtask.

// This replaces the RunSinglePlayout that was previously defined in the .h file.
void GumbelSearch::RunSinglePlayout(classic::Node* start_node) {
    std::vector<classic::Node*> path;
    path.push_back(start_node);
    classic::Node* current_node = start_node;

    while (!current_node->IsTerminal() && node_data_[current_node].is_expanded) {
        if (stop_requested_.load()) break;
        classic::Node* child = SelectChild(current_node);
        if (!child) {
            // No child selected, could be an issue or unexpanded internal node
            break;
        }
        path.push_back(child);
        current_node = child;
    }

    EvalResult eval;
    if (!stop_requested_.load()) {
        if (current_node->IsTerminal()) { // Game-theoretic terminal
             eval = {current_node->GetQ(1), current_node->GetD(), 0.0f, {}}; // Simplified
        } else { // Node needs evaluation or was not expanded by SelectChild path
             eval = ExpandAndGetEvaluation(current_node);
        }
    } else {
        // Search stopped, return neutral eval or handle appropriately
        eval = {0.0f, 0.0f, 0.0f, {}};
    }

    // The value in EvalResult is from the perspective of the player to move at 'current_node'.
    // BackupValue needs to handle perspective changes.
    BackupValue(path, eval.q);
    nodes_visited_count_++; // Count valid playouts
}


} // namespace lczero
