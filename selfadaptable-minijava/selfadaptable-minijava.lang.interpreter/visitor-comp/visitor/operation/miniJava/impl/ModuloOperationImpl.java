package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.IntegerValue;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.Modulo;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;
import minijava.MathService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.ModuloOperation;

public class ModuloOperationImpl extends ExpressionOperationImpl implements ModuloOperation {
	private final Modulo it;

	private final VisitorInterface vis;

	public ModuloOperationImpl(Modulo it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		Value left = ((Value) (((ExpressionOperation)this.it.getLeft().accept(vis)).evaluateExpression((State) (state))));
		Value right = ((Value) (((ExpressionOperation)this.it.getRight().accept(vis)).evaluateExpression((State) (state))));
		if(left instanceof IntegerValue) {
			if(right instanceof IntegerValue) {
				IntegerValue bleft = ((IntegerValue) (left));
				IntegerValue bright = ((IntegerValue) (right));
				IntegerValue tmp = ((IntegerValue) (MiniJavaFactory.eINSTANCE.createIntegerValue()));
				tmp.setValue(MathService.mod((Modulo) (this.it), (int) (bleft.getValue()), (int) (bright.getValue())));
				result = ((IntegerValue) (tmp));
			}
			else {
				result = ((Value) (null));
			}
		}
		else {
			result = ((Value) (null));
		}
		return result;
	}
}
