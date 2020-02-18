package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.Field;
import miniJava.visitor.miniJava.FieldAccess;
import miniJava.visitor.miniJava.FieldBinding;
import miniJava.visitor.miniJava.ObjectInstance;
import miniJava.visitor.miniJava.ObjectRefValue;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;
import org.eclipse.emf.ecoretools.ale.compiler.lib.CollectionService;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.FieldAccessOperation;

public class FieldAccessOperationImpl extends ExpressionOperationImpl implements FieldAccessOperation {
	private final FieldAccess it;

	private final VisitorInterface vis;

	public FieldAccessOperationImpl(FieldAccess it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		ObjectRefValue tmp0 = ((ObjectRefValue) (((ExpressionOperation)this.it.getReceiver().accept(vis)).evaluateExpression((State) (state))));
		ObjectInstance realReceiver = ((ObjectInstance) (tmp0.getInstance()));
		Field fld = ((Field) (this.it.getField()));
		FieldBinding fb = ((FieldBinding) (CollectionService.head(CollectionService.select(realReceiver.getFieldbindings(), (x) -> EqualService.equals((x.getField()), (fld))))));
		result = ((Value) (fb.getValue()));
		return result;
	}
}
