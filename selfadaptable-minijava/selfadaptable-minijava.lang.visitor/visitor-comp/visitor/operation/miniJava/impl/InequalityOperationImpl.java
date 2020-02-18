package visitor.operation.miniJava.impl;

import java.lang.Boolean;
import miniJava.visitor.miniJava.BooleanValue;
import miniJava.visitor.miniJava.Inequality;
import miniJava.visitor.miniJava.IntegerValue;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.NullValue;
import miniJava.visitor.miniJava.ObjectRefValue;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.StringValue;
import miniJava.visitor.miniJava.Value;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.InequalityOperation;

public class InequalityOperationImpl extends ExpressionOperationImpl implements InequalityOperation {
	private final Inequality it;

	private final VisitorInterface vis;

	public InequalityOperationImpl(Inequality it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		Value left = ((Value) (((ExpressionOperation)this.it.getLeft().accept(vis)).evaluateExpression((State) (state))));
		Value right = ((Value) (((ExpressionOperation)this.it.getRight().accept(vis)).evaluateExpression((State) (state))));
		boolean tmp = ((boolean) (false));
		if(left instanceof IntegerValue) {
			if(right instanceof IntegerValue) {
				IntegerValue ileft = ((IntegerValue) (left));
				IntegerValue iright = ((IntegerValue) (right));
				tmp = ((Boolean) (EqualService.equals((ileft.getValue()), (iright.getValue()))));
			}
		}
		else {
			if(left instanceof StringValue) {
				if(right instanceof StringValue) {
					StringValue ileft = ((StringValue) (left));
					StringValue iright = ((StringValue) (right));
					tmp = ((Boolean) (EqualService.equals((ileft.getValue()), (iright.getValue()))));
				}
			}
			else {
				if(left instanceof BooleanValue) {
					if(right instanceof BooleanValue) {
						BooleanValue ileft = ((BooleanValue) (left));
						BooleanValue iright = ((BooleanValue) (right));
						tmp = EqualService.equals((ileft.isValue()), (iright.isValue()));
					}
				}
				else {
					if(left instanceof NullValue) {
						if(right instanceof NullValue) {
							tmp = ((Boolean) (true));
						}
					}
					else {
						if(left instanceof ObjectRefValue) {
							if(right instanceof ObjectRefValue) {
								ObjectRefValue ileft = ((ObjectRefValue) (left));
								ObjectRefValue iright = ((ObjectRefValue) (right));
								tmp = ((Boolean) (EqualService.equals((ileft.getInstance()), (iright.getInstance()))));
							}
						}
					}
				}
			}
		}
		BooleanValue tmpo = ((BooleanValue) (MiniJavaFactory.eINSTANCE.createBooleanValue()));
		tmpo.setValue(!(tmp));
		result = ((BooleanValue) (tmpo));
		return result;
	}
}
