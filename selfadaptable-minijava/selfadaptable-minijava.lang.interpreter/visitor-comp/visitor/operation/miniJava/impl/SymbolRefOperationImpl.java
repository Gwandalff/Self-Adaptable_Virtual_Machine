package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Symbol;
import miniJava.visitor.miniJava.SymbolRef;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ContextOperation;
import visitor.operation.miniJava.StateOperation;
import visitor.operation.miniJava.SymbolRefOperation;
import visitor.operation.miniJava.ValueOperation;

public class SymbolRefOperationImpl extends ExpressionOperationImpl implements SymbolRefOperation {
	private final SymbolRef it;

	private final VisitorInterface vis;

	public SymbolRefOperationImpl(SymbolRef it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		result = ((Value) (((ValueOperation)((ContextOperation)((StateOperation)state.accept(vis)).findCurrentContext().accept(vis)).findBinding((Symbol) (this.it.getSymbol())).getValue().accept(vis)).copyj()));
		return result;
	}
}
