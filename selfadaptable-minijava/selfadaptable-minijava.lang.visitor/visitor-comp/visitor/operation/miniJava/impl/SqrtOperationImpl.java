package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.IntegerValue;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.Sqrt;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;
import minijava.MathService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.SqrtOperation;

public class SqrtOperationImpl extends ExpressionOperationImpl implements SqrtOperation {
	private final Sqrt it;

	private final VisitorInterface vis;

	public SqrtOperationImpl(Sqrt it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		Value expression = ((Value) (((ExpressionOperation)this.it.getExpression().accept(vis)).evaluateExpression((State) (state))));
		if(expression instanceof IntegerValue) {
			IntegerValue number = ((IntegerValue) (expression));
			IntegerValue tmp = ((IntegerValue) (MiniJavaFactory.eINSTANCE.createIntegerValue()));
			tmp.setValue(MathService.sqrt((Sqrt) (this.it), (int) (number.getValue())));
			result = ((IntegerValue) (tmp));
		}
		else {
			result = ((Value) (null));
		}
		return result;
	}
}
