package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.ArrayAccess;
import miniJava.visitor.miniJava.ArrayInstance;
import miniJava.visitor.miniJava.ArrayRefValue;
import miniJava.visitor.miniJava.Assignee;
import miniJava.visitor.miniJava.Assignment;
import miniJava.visitor.miniJava.Context;
import miniJava.visitor.miniJava.Field;
import miniJava.visitor.miniJava.FieldAccess;
import miniJava.visitor.miniJava.FieldBinding;
import miniJava.visitor.miniJava.IntegerValue;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.ObjectInstance;
import miniJava.visitor.miniJava.ObjectRefValue;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Symbol;
import miniJava.visitor.miniJava.SymbolBinding;
import miniJava.visitor.miniJava.SymbolRef;
import miniJava.visitor.miniJava.Value;
import miniJava.visitor.miniJava.VariableDeclaration;
import org.eclipse.emf.ecoretools.ale.compiler.lib.CollectionService;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.AssignmentOperation;
import visitor.operation.miniJava.ContextOperation;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.StateOperation;

public class AssignmentOperationImpl extends StatementOperationImpl implements AssignmentOperation {
	private final Assignment it;

	private final VisitorInterface vis;

	public AssignmentOperationImpl(Assignment it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public void evaluateStatement(State state) {
		Context context = ((Context) (((StateOperation)state.accept(vis)).findCurrentContext()));
		Value right = ((Value) (((ExpressionOperation)this.it.getValue().accept(vis)).evaluateExpression((State) (state))));
		Assignee assignee = ((Assignee) (this.it.getAssignee()));
		if(assignee instanceof SymbolRef) {
			SymbolRef assigneeSymbolRef = ((SymbolRef) (assignee));
			SymbolBinding existingBinding = ((SymbolBinding) (((ContextOperation)context.accept(vis)).findBinding((Symbol) (assigneeSymbolRef.getSymbol()))));
			existingBinding.setValue(right);
		}
		else {
			if(assignee instanceof VariableDeclaration) {
				VariableDeclaration assigneeVariableDeclaration = ((VariableDeclaration) (assignee));
				SymbolBinding binding = ((SymbolBinding) (MiniJavaFactory.eINSTANCE.createSymbolBinding()));
				binding.setSymbol(assigneeVariableDeclaration);
				binding.setValue(right);
				context.getBindings().add(binding);
			}
			else {
				if(assignee instanceof FieldAccess) {
					FieldAccess assigneeFieldAccess = ((FieldAccess) (assignee));
					Field f = ((Field) (assigneeFieldAccess.getField()));
					ObjectRefValue realReceiverValue = ((ObjectRefValue) (((ObjectRefValue) (((ExpressionOperation)assigneeFieldAccess.getReceiver().accept(vis)).evaluateExpression((State) (state))))));
					ObjectInstance realReceiver = ((ObjectInstance) (realReceiverValue.getInstance()));
					FieldBinding existingBinding = ((FieldBinding) (CollectionService.head(CollectionService.select(realReceiver.getFieldbindings(), (x) -> EqualService.equals((x.getField()), (f))))));
					if(EqualService.equals((existingBinding), (null))) {
						FieldBinding binding = ((FieldBinding) (MiniJavaFactory.eINSTANCE.createFieldBinding()));
						binding.setField(f);
						binding.setValue(right);
						realReceiver.getFieldbindings().add(binding);
					}
					else {
						existingBinding.setValue(right);
					}
				}
				else {
					if(assignee instanceof ArrayAccess) {
						ArrayAccess assigneeArrayAccess = ((ArrayAccess) (assignee));
						ArrayRefValue arrayRefValue = ((ArrayRefValue) (((ArrayRefValue) (((ExpressionOperation)assigneeArrayAccess.getObject().accept(vis)).evaluateExpression((State) (state))))));
						ArrayInstance array = ((ArrayInstance) (arrayRefValue.getInstance()));
						IntegerValue integerValue = ((IntegerValue) (((IntegerValue) (((ExpressionOperation)assigneeArrayAccess.getIndex().accept(vis)).evaluateExpression((State) (state))))));
						int index = ((int) (integerValue.getValue()));
						CollectionService.set(array.getValue(), index, right);
					}
					else {
					}
				}
			}
		}
	}
}
