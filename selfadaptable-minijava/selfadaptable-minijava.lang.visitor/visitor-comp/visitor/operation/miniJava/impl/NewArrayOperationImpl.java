package visitor.operation.miniJava.impl;

import java.lang.Integer;
import miniJava.visitor.miniJava.ArrayInstance;
import miniJava.visitor.miniJava.ArrayRefValue;
import miniJava.visitor.miniJava.BooleanTypeRef;
import miniJava.visitor.miniJava.BooleanValue;
import miniJava.visitor.miniJava.ClassRef;
import miniJava.visitor.miniJava.IntegerTypeRef;
import miniJava.visitor.miniJava.IntegerValue;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.NewArray;
import miniJava.visitor.miniJava.NullValue;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.StringTypeRef;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.NewArrayOperation;
import visitor.operation.miniJava.ValueOperation;

public class NewArrayOperationImpl extends ExpressionOperationImpl implements NewArrayOperation {
	private final NewArray it;

	private final VisitorInterface vis;

	public NewArrayOperationImpl(NewArray it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		ArrayInstance res = ((ArrayInstance) (MiniJavaFactory.eINSTANCE.createArrayInstance()));
		IntegerValue sizeArray = ((IntegerValue) (((IntegerValue) (((ExpressionOperation)this.it.getSize().accept(vis)).evaluateExpression((State) (state))))));
		res.setSize(sizeArray.getValue());
		state.getArraysHeap().add(res);
		Value defaultValue = ((Value) (null));
		if(this.it.getType() instanceof IntegerTypeRef) {
			IntegerValue idv = ((IntegerValue) (MiniJavaFactory.eINSTANCE.createIntegerValue()));
			idv.setValue(0);
			defaultValue = ((IntegerValue) (idv));
		}
		else {
			if(this.it.getType() instanceof BooleanTypeRef) {
				BooleanValue idv = ((BooleanValue) (MiniJavaFactory.eINSTANCE.createBooleanValue()));
				idv.setValue(false);
				defaultValue = ((BooleanValue) (idv));
			}
			else {
				if(this.it.getType() instanceof StringTypeRef) {
					NullValue idv = ((NullValue) (MiniJavaFactory.eINSTANCE.createNullValue()));
					defaultValue = ((NullValue) (idv));
				}
				else {
					if(this.it.getType() instanceof ClassRef) {
						NullValue idv = ((NullValue) (MiniJavaFactory.eINSTANCE.createNullValue()));
						defaultValue = ((NullValue) (idv));
					}
				}
			}
		}
		int i = ((int) (0));
		int sz = ((int) (res.getSize()));
		while ((i) < (sz)) {
			Value dv = ((Value) (defaultValue));
			Value v = ((Value) (((ValueOperation)dv.accept(vis)).copyj()));
			res.getValue().add(v);
			i = ((Integer) ((i) + (1)));
		}
		ArrayRefValue ret = ((ArrayRefValue) (MiniJavaFactory.eINSTANCE.createArrayRefValue()));
		ret.setInstance(res);
		result = ((ArrayRefValue) (ret));
		return result;
	}
}
