package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.BoolConstant;
import miniJava.visitor.miniJava.BooleanValue;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.BoolConstantOperation;

public class BoolConstantOperationImpl extends ExpressionOperationImpl implements BoolConstantOperation {
	private final BoolConstant it;

	private final VisitorInterface vis;

	public BoolConstantOperationImpl(BoolConstant it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		BooleanValue ret = ((BooleanValue) (MiniJavaFactory.eINSTANCE.createBooleanValue()));
		ret.setValue(EqualService.equals((this.it.getValue()), ("true")));
		result = ((BooleanValue) (ret));
		return result;
	}
}
