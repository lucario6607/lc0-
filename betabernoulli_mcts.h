#ifndef BETABERNOULLI_MCTS_H
#define BETABERNOULLI_MCTS_H

#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iostream> // For print_tree_stats and run_simulation_example

namespace BetaMCTS {

// Forward declaration
class BetaBernoulliNode;

class BetaBernoulliMCTS {
public:
    std::shared_ptr<BetaBernoulliNode> root_;
    std::mt19937 random_generator_;
    double exploration_constant_ = 1.0; // Corresponds to C in UCB1

    BetaBernoulliMCTS(int seed = 12345) : random_generator_(seed) {
        root_ = std::make_shared<BetaBernoulliNode>();
    }

    std::shared_ptr<BetaBernoulliNode> select_node(std::shared_ptr<BetaBernoulliNode> node) {
        while (!node->children_.empty()) {
            node = *std::max_element(node->children_.begin(), node->children_.end(),
                [](const auto& a, const auto& b) {
                    return a->ucb1(1.0) < b->ucb1(1.0); // Assuming exploration_constant_ is 1.0 for now
                });
        }
        return node;
    }

    void expand_node(std::shared_ptr<BetaBernoulliNode> node, const std::vector<double>& priors) {
        if (node->children_.empty() && !priors.empty()) {
            for (size_t i = 0; i < priors.size(); ++i) {
                auto child_node = std::make_shared<BetaBernoulliNode>();
                child_node->parent_ = node;
                // child_node->prior_ = priors[i]; // Prior is not directly used in BetaBernoulliNode like this
                node->children_.push_back(child_node);
            }
        }
    }

    double simulate(std::shared_ptr<BetaBernoulliNode> node) {
        // In a real MCTS, this would be a rollout or a network evaluation.
        // For BetaBernoulli, the "simulation" is implicitly handled by the update step (alpha/beta updates).
        // We return a random result here as a placeholder, actual value comes from game result or model.
        std::uniform_real_distribution<> dist(0.0, 1.0);
        return dist(random_generator_);
    }

    void backpropagate(std::shared_ptr<BetaBernoulliNode> node, double result) {
        while (node) {
            node->update(result);
            node = node->parent_.lock();
        }
    }

    void run_simulation(int num_simulations, const std::vector<double>& initial_priors = {}) {
        for (int i = 0; i < num_simulations; ++i) {
            std::shared_ptr<BetaBernoulliNode> node = select_node(root_);
            if (node->get_n() == 0 && !initial_priors.empty()) { // Only expand with initial_priors if it's the root and unvisited.
                 // This logic might need adjustment depending on when/how priors are available.
                 // If priors are for the children of the selected node:
                if (node == root_ ) { // typically expansion happens from a new node
                    expand_node(node, initial_priors);
                    // After expansion, selection might pick one of the new children or stay.
                    // If children were added, select one to simulate.
                    // This simple MCTS will simulate from the expanded node itself.
                    // A more complex MCTS might select a child before simulation.
                }
            } else if (node->get_n() > 0 && node->children_.empty() && node != root_) {
                // If not root, and visited, but no children, we might expand it based on some policy
                // For simplicity, let's assume we always try to expand with a default prior if not root
                // This is a placeholder for a more sophisticated expansion strategy
                std::vector<double> default_priors = {0.5, 0.5}; // Example: binary action space
                expand_node(node, default_priors);
            }


            double result;
            // If the node has children (because it was expanded), we should select one for simulation.
            // For this example, we'll simulate from the selected node `node` itself.
            // In a typical MCTS, if `node` was expanded, you'd pick one of its children to simulate.
            // However, our `simulate` is more of a placeholder. The actual "outcome" for BetaBernoulli
            // comes from the environment (e.g. game win/loss).
            if (!node->children_.empty()){
                 // If children exist, it means we expanded. Let's pick the first child for simulation as a simple strategy.
                 // A proper MCTS would select a child based on UCB or other criteria before simulation,
                 // or the simulation would start from the expanded node `node` if it's terminal or a rollout is done.
                 // Here, we'll just use the result for the expanded node itself.
                 result = simulate(node->children_[0]); // Placeholder: simulate one of the new children
            } else {
                 result = simulate(node); // Simulate from the selected node if no children
            }
            backpropagate(node, result);
        }
    }

    void print_tree_stats(std::shared_ptr<BetaBernoulliNode> node, int indent = 0) {
        if (!node) return;
        std::cout << std::string(indent, ' ') << "Node Stats: N=" << node->get_n()
                  << ", Alpha=" << node->alpha_ << ", Beta=" << node->beta_
                  << ", Value=" << node->value() << ", UCB1=" << node->ucb1(exploration_constant_) << std::endl;
        for (const auto& child : node->children_) {
            print_tree_stats(child, indent + 2);
        }
    }
};

class BetaBernoulliNode {
public:
    std::weak_ptr<BetaBernoulliNode> parent_;
    std::vector<std::shared_ptr<BetaBernoulliNode>> children_;
    double alpha_ = 1.0; // Prior for wins (or positive outcomes)
    double beta_ = 1.0;  // Prior for losses (or negative outcomes)
    // double prior_ = 0.0; // Policy prior (not directly used in alpha/beta updates in this simple version)

    BetaBernoulliNode() = default;

    double get_n() const {
        return alpha_ + beta_ - 2.0; // -2.0 because alpha and beta are initialized to 1
    }

    double value() const {
        if (get_n() == 0) return 0.0; // Or some default value like 0.5
        return alpha_ / (alpha_ + beta_);
    }

    void update(double result) {
        // result is typically 1 for win, 0 for loss
        if (result >= 0.5) { // Assuming result > 0.5 is a "win"
            alpha_++;
        } else {
            beta_++;
        }
    }

    double ucb1(double exploration_constant) const {
        if (get_n() == 0) {
            return std::numeric_limits<double>::max(); // Ensure unvisited nodes are prioritized
        }
        double N_parent = 0;
        if (auto p = parent_.lock()) {
            N_parent = p->get_n();
        }
        if (N_parent == 0) N_parent = get_n(); // If no parent or parent not visited, use own N (e.g. for root)


        return value() + exploration_constant * std::sqrt(std::log(N_parent) / get_n());
    }
};


// Dummy LeelaIntegration class for placeholder
class LeelaIntegration {
public:
    LeelaIntegration() {}

    // Placeholder for getting policy priors from Leela Zero
    std::vector<double> get_leela_policy_priors(const std::string& board_state) {
        // In a real scenario, this would query Leela Zero
        // For this example, returning a dummy distribution
        std::cout << "LeelaIntegration: Querying priors for state (dummy): " << board_state << std::endl;
        return {0.4, 0.3, 0.3}; // Example: 3 possible moves
    }

    // Placeholder for getting value from Leela Zero
    double get_leela_value(const std::string& board_state) {
        // In a real scenario, this would query Leela Zero
        std::cout << "LeelaIntegration: Querying value for state (dummy): " << board_state << std::endl;
        return 0.5; // Dummy value
    }

    // Example function to show how MCTS might interact with Leela
    void run_simulation_example() {
        BetaBernoulliMCTS mcts_instance(5678); // Different seed for this example
        std::string current_board_state = "initial_state";

        // Get priors for the root state from Leela
        std::vector<double> initial_priors = get_leela_policy_priors(current_board_state);
        
        // Expand root node with these priors before starting simulations
        // Note: The current BetaBernoulliMCTS `expand_node` doesn't use priors directly to set child node priors.
        // It just creates children. A more integrated version might pass priors to children.
        // For now, we pass it to run_simulation which might use it (as per current run_simulation logic)
        // mcts_instance.expand_node(mcts_instance.root_, initial_priors);


        std::cout << "Running MCTS simulations with Leela-like integration..." << std::endl;
        mcts_instance.run_simulation(100, initial_priors); // Run 100 simulations

        std::cout << "Tree statistics after simulations:" << std::endl;
        mcts_instance.print_tree_stats(mcts_instance.root_);

        // Example of selecting the best move after simulations
        if (!mcts_instance.root_->children_.empty()) {
            auto best_child = *std::max_element(mcts_instance.root_->children_.begin(), mcts_instance.root_->children_.end(),
                [](const auto& a, const auto& b) {
                    return a->get_n() < b->get_n(); // Select child with most visits
                });
            std::cout << "Best child to select has N=" << best_child->get_n() 
                      << " and value=" << best_child->value() << std::endl;
        } else {
            std::cout << "Root has no children after simulation." << std::endl;
        }
    }
};

} // namespace BetaMCTS

#endif // BETABERNOULLI_MCTS_H
